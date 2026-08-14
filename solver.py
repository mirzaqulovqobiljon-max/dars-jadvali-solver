"""
Dars jadvali generatsiyasi — Google OR-Tools CP-SAT solver.

Kirish (JSON dict):
{
  "school": {"daysPerWeek": 6, "lessonsPerDay": 7},
  "teachers": [{"id","name","methodicalDay"(int|null),"maxHours"(int),"unavailable":{"d-s":true}}],
  "subjects": [{"id","name","difficulty"(1-10),"unavailable":{"d-s":true}}],
  "classes":  [{"id","name"}],
  "rooms":    [{"id","name"}],
  "assignments":[{"id","classId","subjectId","teacherId","hoursPerWeek"(int),
                  "roomId"(id|null),"isSplit"(bool),"splitTeacherId"(id|null),"splitRoomId"(id|null)}]
}

Chiqish (JSON dict):
{
  "entries":[{"classId","subjectId","teacherId","roomId","day","lesson","group"(null|"A"|"B")}],
  "unfilled":[{"classId","subjectId","teacherId","hours"}],
  "status": "OPTIMAL"|"FEASIBLE"|"INFEASIBLE",
  "relaxed": 0|1|2,        # qaysi bosqichda yechim topildi (0 = to'liq model)
  "warnings": ["..."],     # yumshatilgan cheklovlar haqida ogohlantirish
  "stats": {...}
}

Cheklovlar:
  HARD (buzilmaydi):
   - sinf bir vaqtda bitta dars
   - o'qituvchi bir vaqtda bitta dars (parallel yo'q), split ikkinchi o'qituvchi ham
   - metodik kun / o'qituvchi bandligi / fan kun-soat cheklovi
   - xona bir vaqtda bitta dars
   - bir kunda bir xil fan ko'pi bilan 1 marta (per class; agar fanning haftalik
     soati kunlar sonidan ko'p bo'lsa — 2 marta)
   - o'qituvchi haftalik maksimal soati
   - BO'SH OYNA YO'Q: har sinf har kunda darslar 0-slotdan ketma-ket (prefix)
   - IKKI SMENA: har sinfning o'z smenasi bor; o'qituvchi/xona bandligi slot
     RAQAMI bo'yicha emas, REAL VAQT bo'yicha tekshiriladi (1-smena 1-soati
     08:00, 2-smena 1-soati 14:00 — kesishmaydi)
  SOFT (optimallashtiriladi):
   - imkon qadar ko'p dars joylashtirilsin (placed maksimal)
   - kunlar muvozanati (bir kun juda ko'p / juda kam bo'lmasin)
   - og'ir fanlar ertaroq soatlarga
"""
from ortools.sat.python import cp_model

# Versiya belgisi — klient serverda qaysi kod turganini shu orqali aniqlaydi.
# "entry" maydoni esa API qaysi funksiyani chaqirganini ko'rsatadi:
#   solve_safe      -> to'g'ri (INFEASIBLE kaskadi va tekis taqsimot ishlaydi)
#   solve_timetable -> NOTO'G'RI (yumshatish kaskadi ishlamaydi)
ENGINE_VERSION = "v2-2026-08"


def diagnose(data):
    """Jadval to'liq chiqmasligining ANIQ sabablarini topadi (o'zbekcha xabarlar).
    Bu — matematik tekshiruv: hech qanday dastur bu holatlarni joylashtira olmaydi."""
    school = data["school"]
    days = int(school["daysPerWeek"])
    slots = int(school["lessonsPerDay"])
    # Har sinfning smenasidagi dars soni har xil bo'lishi mumkin
    _cshift = {c["id"]: int(c.get("shift") or 1) for c in data["classes"]}
    _sslots = {sh: shift_cfg(school, sh)["lessons"] for sh in (1, 2)}

    def _cls_total_slots(cid):
        bd = next((c.get("busyDays") or {} for c in data["classes"] if c["id"] == cid), {})
        ish_kun = days - len([k for k, v in bd.items() if v])
        return max(1, ish_kun) * _sslots.get(_cshift.get(cid, 1), slots)

    total_slots = days * slots
    teachers = {t["id"]: t for t in data["teachers"]}
    msgs = []

    # 1) SINF: haftalik slotdan ko'p dars biriktirilganmi?
    cls_names = {c["id"]: c.get("name", c["id"]) for c in data["classes"]}
    cls_load = {}
    for a in data["assignments"]:
        cls_load[a["classId"]] = cls_load.get(a["classId"], 0) + int(a.get("hoursPerWeek") or 0)
    for cid, load in cls_load.items():
        total_slots = _cls_total_slots(cid)     # shu sinfning haqiqiy joy soni
        if load > total_slots:
            msgs.append(
                f"SINF {cls_names.get(cid, cid)}: {load} soat biriktirilgan, lekin haftada "
                f"{total_slots} ta joy bor ({days} kun × {slots} soat) — {load - total_slots} soati "
                f"hech qachon sig'maydi. Yechim: bu sinf soatlarini kamaytiring."
            )

    # 2) O'QITUVCHI: bo'sh vaqti / max soatidan ko'p yuk berilganmi?
    t_load = {}
    for a in data["assignments"]:
        h = int(a.get("hoursPerWeek") or 0)
        t_load[a["teacherId"]] = t_load.get(a["teacherId"], 0) + h
        if a.get("isSplit") and a.get("splitTeacherId"):
            t_load[a["splitTeacherId"]] = t_load.get(a["splitTeacherId"], 0) + h
    for tid, load in t_load.items():
        t = teachers.get(tid)
        if not t:
            msgs.append(f"DIQQAT: o'chirilgan o'qituvchiga {load} soat biriktirilgan — "
                        f"'Dars biriktirish'da bu qatorlarni tahrirlang.")
            continue
        avail = 0
        for d in range(days):
            if t.get("methodicalDay") is not None and int(t["methodicalDay"]) == d:
                continue
            for s in range(slots):
                if t.get("unavailable", {}).get(f"{d}-{s}"):
                    continue
                avail += 1
        mx = t.get("maxHours")
        cap = min(avail, int(mx)) if mx else avail
        if load > cap:
            if mx and int(mx) < avail:
                sabab = f"'max soat' cheklovi {mx} qilib qo'yilgan (uni oshiring)"
            else:
                sabab = (f"metodik kuni va band soatlaridan keyin haftada faqat {avail} ta "
                         f"bo'sh joyi qoladi")
            msgs.append(
                f"O'QITUVCHI {t.get('name', tid)}: {load} soat biriktirilgan, lekin ko'pi bilan "
                f"{cap} soat bera oladi — sabab: {sabab}. Kamida {load - cap} soati joylashmaydi. "
                f"Yechim: bu fanga ikkinchi o'qituvchi qo'shing yoki cheklovni yumshating."
            )

    return msgs


def _greedy_seed(data, days, slots, teachers, subjects, assignments):
    """Tez greedy jadval: barcha qattiq shartlarga rioya qiladi, zich va muvozanatli.
    Natija: set of (ai, d, s) — CP-SAT uchun boshlang'ich yechim (hint)."""
    t_busy = set()      # (tid, d, s)
    r_busy = set()      # (rid, d, s)
    c_busy = set()      # (cid, d, s)
    csd = set()         # (cid, subid, d) — bir kunda bir fan
    t_hours = {}        # tid -> jami soat
    c_dayload = {}      # (cid, d) -> soat soni

    def teacher_ok(tid, d, s):
        t = teachers.get(tid)
        if not t:
            return False
        if t.get("methodicalDay") is not None and int(t["methodicalDay"]) == d:
            return False
        if t.get("unavailable", {}).get(f"{d}-{s}"):
            return False
        return True

    def subject_ok(subid, d, s):
        su = subjects.get(subid)
        return not (su and su.get("unavailable", {}).get(f"{d}-{s}"))

    def teacher_cap_ok(tid, extra=1):
        t = teachers.get(tid, {})
        mx = t.get("maxHours")
        if not mx:
            return True
        return t_hours.get(tid, 0) + extra <= int(mx)

    # Har biriktirishni soatlarga yoyamiz; eng cheklangan (qiyin) o'qituvchilar avval
    tasks = []
    for ai, a in enumerate(assignments):
        for _ in range(int(a.get("hoursPerWeek") or 0)):
            tasks.append((ai, a))

    def constraint_score(item):
        ai, a = item
        t = teachers.get(a["teacherId"], {})
        load = sum(int(b.get("hoursPerWeek") or 0) for b in assignments
                   if b["teacherId"] == a["teacherId"])
        meth = 1 if t.get("methodicalDay") is not None else 0
        unav = len([1 for v in (t.get("unavailable") or {}).values() if v])
        return -(load * 2 + meth * slots + unav)  # og'ir yuklangan avval

    tasks.sort(key=constraint_score)

    seed = set()
    for ai, a in enumerate_tasks(tasks):
        cid, subid, tid = a["classId"], a["subjectId"], a["teacherId"]
        stid = a.get("splitTeacherId") if a.get("isSplit") else None
        best = None
        best_score = None
        for d in range(days):
            if (cid, subid, d) in csd:
                continue
            for s in range(slots):
                if (cid, d, s) in c_busy:
                    continue
                if (tid, d, s) in t_busy or not teacher_ok(tid, d, s):
                    continue
                if stid and ((stid, d, s) in t_busy or not teacher_ok(stid, d, s)):
                    continue
                if not subject_ok(subid, d, s):
                    continue
                if a.get("roomId") and (a["roomId"], d, s) in r_busy:
                    continue
                # ball: zichlik (slot == shu kundagi darslar soni -> oynasiz) + muvozanat
                dl = c_dayload.get((cid, d), 0)
                sc = -abs(s - dl) * 10 - dl * 3 - s
                if best_score is None or sc > best_score:
                    best_score = sc
                    best = (d, s)
        if best is None:
            continue
        if not teacher_cap_ok(tid) or (stid and not teacher_cap_ok(stid)):
            continue
        d, s = best
        seed.add((ai, d, s))
        c_busy.add((cid, d, s))
        t_busy.add((tid, d, s))
        if stid:
            t_busy.add((stid, d, s))
        if a.get("roomId"):
            r_busy.add((a["roomId"], d, s))
        csd.add((cid, subid, d))
        t_hours[tid] = t_hours.get(tid, 0) + 1
        if stid:
            t_hours[stid] = t_hours.get(stid, 0) + 1
        c_dayload[(cid, d)] = c_dayload.get((cid, d), 0) + 1
    return seed


def enumerate_tasks(tasks):
    """(ai, a) juftliklarini qaytaradi (tasks allaqachon (ai,a) formatida)."""
    for ai, a in tasks:
        yield ai, a


def _num(v, default):
    """Bo'sh/None qiymatlar uchun standart qiymat qaytaradi."""
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def shift_cfg(school, shift):
    """Smena sozlamalari — frontend'dagi shiftCfg() bilan AYNAN bir xil.
    2-smenaning o'z boshlanish vaqti, dars soni va davomiyligi bo'lishi mumkin."""
    if int(shift or 1) == 2:
        return {
            "start": school.get("startShift2") or "14:00",
            "lessons": _num(school.get("lessonsPerDay2"), _num(school.get("lessonsPerDay"), 6)),
            "dur": _num(school.get("lessonDuration2"), _num(school.get("lessonDuration"), 45)),
            "brk": _num(school.get("breakDuration2"), _num(school.get("breakDuration"), 10)),
            "big_after": _num(school.get("bigBreakAfter2"), _num(school.get("bigBreakAfter"), 0)),
            "big_dur": _num(school.get("bigBreakDuration2"), _num(school.get("bigBreakDuration"), 0)),
        }
    return {
        "start": school.get("startShift1") or "08:00",
        "lessons": _num(school.get("lessonsPerDay"), 6),
        "dur": _num(school.get("lessonDuration"), 45),
        "brk": _num(school.get("breakDuration"), 10),
        "big_after": _num(school.get("bigBreakAfter"), 0),
        "big_dur": _num(school.get("bigBreakDuration"), 0),
    }


def _hhmm(t):
    try:
        parts = str(t or "0:0").split(":")
        return int(parts[0] or 0) * 60 + int(parts[1] or 0)
    except (ValueError, IndexError):
        return 0


def lesson_span(school, shift, i):
    """i-darsning (boshlanish, tugash) daqiqasi — kun boshidan hisoblab."""
    c = shift_cfg(school, shift)
    t = _hhmm(c["start"])
    for k in range(i):
        t += c["dur"] + c["brk"]
        if c["big_after"] and k + 1 == c["big_after"]:
            t += (c["big_dur"] - c["brk"])
    return (t, t + c["dur"])


def slots_overlap(school, sh1, i1, sh2, i2):
    """Ikki (smena, soat) REAL VAQT bo'yicha kesishadimi.
    1-smena 1-soati (08:00) va 2-smena 1-soati (14:00) — kesishmaydi,
    shuning uchun bitta o'qituvchi ikkalasida ham dars bera oladi."""
    a = lesson_span(school, sh1, i1)
    b = lesson_span(school, sh2, i2)
    return a[0] < b[1] and b[0] < a[1]


def solve_timetable(data, max_seconds=20, relax_days=0):
    """relax_days — kunlik yuk cheklovlarining yumshatish darajasi:
         0 = ideal tekis taqsimot (floor..ceil)
         1 = +-1 zaxira
         2 = chegara yo'q
       INFEASIBLE holatida solve_safe() bu darajalarni ketma-ket sinaydi."""
    school = data["school"]
    days = int(school["daysPerWeek"])
    slots = int(school["lessonsPerDay"])
    # ===== SMENALAR =====
    # Har sinfning o'z smenasi bor; smenalar dars soni va vaqti bilan farq qiladi.
    class_shift = {c["id"]: int(c.get("shift") or 1) for c in data["classes"]}
    shift_slots = {sh: shift_cfg(school, sh)["lessons"] for sh in (1, 2)}
    # Modelning umumiy slot diapazoni — ikki smenadagi eng kattasi
    slots = max(slots, shift_slots.get(1, slots), shift_slots.get(2, 0))

    def cls_slots(cid):
        """Shu sinfda kuniga nechta dars soati bor."""
        return shift_slots.get(class_shift.get(cid, 1), slots)

    # Vaqt jihatdan kesishuvchi (smena, soat) juftliklari — oldindan hisoblanadi
    _ov_cache = {}

    def time_conflict(cid1, s1, cid2, s2):
        sh1, sh2 = class_shift.get(cid1, 1), class_shift.get(cid2, 1)
        if sh1 == sh2:
            return s1 == s2
        key = (sh1, s1, sh2, s2)
        if key not in _ov_cache:
            _ov_cache[key] = slots_overlap(school, sh1, s1, sh2, s2)
        return _ov_cache[key]

    # O'qituvchi bandligini VAQT bo'yicha guruhlash uchun: (smena,soat) -> vaqt oynasi kaliti
    def time_cells(cid, s):
        """Shu (sinf, soat) qaysi vaqt oynalarini egallaydi — ikkala smena bo'yicha."""
        sh = class_shift.get(cid, 1)
        cells = [(sh, s)]
        other = 2 if sh == 1 else 1
        for j in range(shift_slots.get(other, 0)):
            if slots_overlap(school, sh, s, other, j):
                cells.append((other, j))
        return cells
    teachers = {t["id"]: t for t in data["teachers"]}
    subjects = {s["id"]: s for s in data["subjects"]}
    classes = data["classes"]
    assignments = data["assignments"]
    class_busy = {}   # cid -> set(busy day indexes)
    for c in classes:
        bd = c.get("busyDays") or {}
        class_busy[c["id"]] = set(int(k) for k, v in bd.items() if v)

    # ===== QULFLANGAN SINFLAR (fixedEntries) =====
    # Qulflangan sinf darslari o'z joyida qat'iy qoladi. Ular optimizatsiyaga
    # kirmaydi — solver faqat ochiq sinflarni tuzadi, lekin qulflangan darslar
    # o'qituvchi/xona bandligiga ta'sir qiladi (to'qnashuv bo'lmasligi uchun).
    fixed_entries = data.get("fixedEntries") or []
    locked_class_ids = set(e["classId"] for e in fixed_entries)
    # qulflangan sinflarga tegishli biriktirishlarni optimizatsiyadan chiqaramiz
    if locked_class_ids:
        assignments = [a for a in assignments if a["classId"] not in locked_class_ids]

    m = cp_model.CpModel()

    D = range(days)
    S = range(slots)

    # qulflangan darslar egallagan (o'qituvchi/xona) slotlar — bandlik
    fixed_teacher = set()
    fixed_room = set()
    fixed_teacher_cell = set()   # (tid, day, (smena,soat)) — vaqt oynasi bo'yicha
    fixed_room_cell = set()
    for e in fixed_entries:
        fixed_teacher.add((e["teacherId"], e["day"], e["lesson"]))
        if e.get("roomId"):
            fixed_room.add((e["roomId"], e["day"], e["lesson"]))
        for cell in time_cells(e["classId"], e["lesson"]):
            fixed_teacher_cell.add((e["teacherId"], e["day"], cell))
            if e.get("roomId"):
                fixed_room_cell.add((e["roomId"], e["day"], cell))

    def teacher_ok(tid, d, s):
        """o'qituvchi (tid) d-kun s-soatda ishlay oladimi (metodik/bandlik)."""
        t = teachers.get(tid)
        if not t:
            return False
        if t.get("methodicalDay") is not None and int(t["methodicalDay"]) == d:
            return False
        if t.get("unavailable", {}).get(f"{d}-{s}"):
            return False
        return True

    def subject_ok(subid, d, s):
        su = subjects.get(subid)
        if su and su.get("unavailable", {}).get(f"{d}-{s}"):
            return False
        return True

    # x[(a_index, d, s)] = 1 -> assignment a shu (d,s) da dars
    x = {}
    for ai, a in enumerate(assignments):
        for d in D:
            for s in S:
                ok = True
                # fan cheklovi
                if not subject_ok(a["subjectId"], d, s):
                    ok = False
                # asosiy o'qituvchi
                if ok and not teacher_ok(a["teacherId"], d, s):
                    ok = False
                # split ikkinchi o'qituvchi ham shu vaqtda ishlashi kerak
                if ok and a.get("isSplit") and a.get("splitTeacherId"):
                    if not teacher_ok(a["splitTeacherId"], d, s):
                        ok = False
                # qulflangan darslar bilan to'qnashuv bo'lmasin (REAL VAQT bo'yicha)
                if ok:
                    cells = set(time_cells(a["classId"], s))
                    if any((a["teacherId"], d, cell) in fixed_teacher_cell for cell in cells):
                        ok = False
                    if ok and a.get("isSplit") and a.get("splitTeacherId"):
                        if any((a["splitTeacherId"], d, cell) in fixed_teacher_cell for cell in cells):
                            ok = False
                    if ok and a.get("roomId"):
                        if any((a["roomId"], d, cell) in fixed_room_cell for cell in cells):
                            ok = False
                # sinfning band kuni — dars qo'yilmaydi
                if ok and d in class_busy.get(a["classId"], set()):
                    ok = False
                # SMENA: bu sinfda bunday soat umuman yo'q (masalan 2-smenada 5 soat)
                if ok and s >= cls_slots(a["classId"]):
                    ok = False
                # BELGILANGAN VAQTLI FAN (subject.fixedSlot) — masalan "Kelajak
                # soati" har doim dushanba 1-soatda. Bu QAT'IY: fan boshqa
                # kun/soatga umuman qo'yilmaydi, shuning uchun o'zgaruvchi ham
                # yaratilmaydi (model kichrayadi, tezlashadi).
                # Ilgari bu cheklov FAQAT klientda bor edi — server uni
                # bilmasdi va CP-SAT fanni istalgan joyga qo'yib yuborardi.
                if ok:
                    fx = _fixed_slot(subjects.get(a["subjectId"]))
                    if fx is not None and (d, s) != fx:
                        ok = False
                if ok:
                    x[(ai, d, s)] = m.NewBoolVar(f"x_{ai}_{d}_{s}")

    # Har assignment ko'pi bilan hours marta joylashadi (kamroq bo'lishi mumkin -> unfilled)
    placed = {}
    for ai, a in enumerate(assignments):
        hours = int(a.get("hoursPerWeek") or 0)
        vars_a = [x[(ai, d, s)] for d in D for s in S if (ai, d, s) in x]
        p = m.NewIntVar(0, hours, f"placed_{ai}")
        if vars_a:
            m.Add(sum(vars_a) == p)
        else:
            m.Add(p == 0)
        m.Add(p <= hours)
        placed[ai] = (p, hours)

    # Sinf bir vaqtda bitta dars
    # y[c,d,s] = shu sinf shu (d,s) da band
    y = {}
    class_ids = [c["id"] for c in classes]
    for c in class_ids:
        for d in D:
            for s in S:
                yv = m.NewBoolVar(f"y_{c}_{d}_{s}")
                y[(c, d, s)] = yv
                occ = [x[(ai, d, s)] for ai, a in enumerate(assignments)
                       if a["classId"] == c and (ai, d, s) in x]
                if occ:
                    m.Add(yv == sum(occ))   # occ<=1 shu bilan ta'minlanadi
                    m.Add(sum(occ) <= 1)
                else:
                    m.Add(yv == 0)

    # ===== QAT'IY: ORALIQDA BO'SH OYNA YO'Q (prefix) =====
    # Agar (d, s+1) band bo'lsa, (d, s) ham band bo'lishi shart -> darslar 1-soatdan
    # ketma-ket, bo'sh joy faqat kun oxirida.
    for c in class_ids:
        n_slots = cls_slots(c)          # shu sinf smenasidagi dars soni
        for d in D:
            for s in range(n_slots - 1):
                m.Add(y[(c, d, s)] >= y[(c, d, s + 1)])
            # smenada mavjud bo'lmagan soatlar umuman band bo'lmasin
            for s in range(n_slots, slots):
                m.Add(y[(c, d, s)] == 0)

    # ===== QAT'IY: har kun 0 YOKI kamida MIN_PER_DAY soat + tekis taqsimot =====
    # DIQQAT: sinfning BAND KUNLARINI (busyDays) hisobga olamiz — ish kunlari kamayadi.
    MIN_PER_DAY = 4
    import math
    for c in class_ids:
        total_c = sum(int(a.get("hoursPerWeek") or 0)
                      for a in assignments if a["classId"] == c)
        busy = class_busy.get(c, set())
        work_days = max(1, days - len([d for d in D if d in busy]))  # haqiqiy ish kunlari
        # min_day: kam darsli sinflarni majburlamaymiz (ular joylashsin, "chala kun" mayli)
        if total_c < MIN_PER_DAY * work_days:
            min_day = 1          # kam dars -> erkin joylashsin
        else:
            min_day = MIN_PER_DAY
        # Agar darslar soni ish kunlariga yetsa (har kunga kamida 1 tadan), HAR ish
        # kunida kamida 1 dars bo'lishini QAT'IY talab qilamiz -> bir kun butunlay
        # bo'sh qolmaydi (masalan Dushanba bo'sh qolmaydi).
        force_all_days = total_c >= work_days
        # ===== KUNLIK YUK CHEGARALARI (bosqichli qattiqlik) =====
        # relax_days — solve_safe() beradigan yumshatish darajasi:
        #   0 = ideal tekis taqsimot (floor..ceil) — bir kun 7, boshqasi 3 bo'lolmaydi
        #   1 = +-1 zaxira (INFEASIBLE bo'lganda)
        #   2 = chegara yo'q (oxirgi chora)
        # Ilgari bu yerda faqat qat'iy `ceil` bor edi: model tez-tez INFEASIBLE
        # bo'lardi. Keyin +1 qo'shilgach INFEASIBLE yo'qoldi-yu, taqsimot buzildi
        # (bir kun 7 dars, boshqasi 3). Bosqichli yondashuv ikkovini ham hal qiladi.
        even_hi = math.ceil(total_c / work_days)
        even_lo = total_c // work_days

        # JUFT DARS ISTISNOSI: agar sinfda doubleLesson fani bo'lsa, kunlik
        # yuqori chegara kamida 2 bo'lishi SHART. Aks holda haftalik soati
        # kunlar sonidan kam fanlarda (masalan 2 soat / 6 kun -> ceil = 1)
        # chegara juft darsni JISMONAN imkonsiz qilib qo'yadi va u ikki
        # kunga ajralib ketadi. (Test 7 aynan shu holatni ushlagan.)
        has_double = any(
            (subjects.get(a["subjectId"]) or {}).get("doubleLesson")
            and int(a.get("hoursPerWeek") or 0) >= 2
            for a in assignments if a["classId"] == c
        )
        if has_double:
            even_hi = max(even_hi, 2)
        if relax_days >= 3:
            # oxirgi chora: hech qanday kunlik chegara yo'q
            hi_day = cls_slots(c)
            lo_day = 0
        elif relax_days == 2:
            # chegara bo'sh, LEKIN kun butunlay bo'sh qolmasin.
            # Busiz [0,0,0,0,7,7] kabi jadval chiqardi — 4 kun bo'sh, 2 kun
            # to'la yuklangan. Maktab uchun bu yaroqsiz.
            hi_day = cls_slots(c)
            lo_day = 1 if force_all_days else 0
        elif relax_days == 1:
            hi_day = min(cls_slots(c), max(min_day, even_hi + 1))
            lo_day = max(0, even_lo - 1)
        else:
            hi_day = min(cls_slots(c), max(min_day, even_hi))
            lo_day = even_lo
        # har kunga majburiy minimum: darslar barcha kunga yetishi shart.
        # total_c ni work_days ga bo'lганда butun qism — har kunга kафolatли minimum.
        forced_min = min(min_day, total_c // work_days) if force_all_days else min_day
        if forced_min < 1:
            forced_min = 1
        for d in D:
            if d in busy:
                continue  # band kun — cheklov qo'ymaymiz (x allaqachon yo'q)
            day_load = sum(y[(c, d, s)] for s in S)
            if force_all_days:
                m.Add(day_load <= hi_day)
                # Quyi chegara: kun juda bo'sh qolmasin. relax_days=2 da lo_day=0,
                # ya'ni cheklov amalda yo'q -> INFEASIBLE xavfi qolmaydi.
                if lo_day >= 1:
                    m.Add(day_load >= lo_day)
            else:
                # kam darsli sinf: kun bo'sh bo'lishi mumkin, lekin dars bo'lsa >=min_day
                has_day = m.NewBoolVar(f"hasday_{c}_{d}")
                m.Add(day_load >= 1).OnlyEnforceIf(has_day)
                m.Add(day_load == 0).OnlyEnforceIf(has_day.Not())
                if not relax_days:
                    m.Add(day_load >= min_day).OnlyEnforceIf(has_day)
                m.Add(day_load <= hi_day)

    # ===== O'QITUVCHI BIR VAQTDA BITTA DARS =====
    # DIQQAT: bandlik slot RAQAMI bo'yicha emas, REAL VAQT bo'yicha tekshiriladi.
    # 1-smena 1-soati (08:00) va 2-smena 1-soati (14:00) — har xil vaqt, ya'ni
    # bitta o'qituvchi ikkalasida ham dars bera oladi. Aksincha, 1-smenaning
    # oxirgi soati 2-smenaning birinchi soati bilan ustma-ust tushishi mumkin —
    # bunda to'qnashuv bo'ladi.
    teacher_slot = {}  # (tid, d, (smena,soat)) -> vars
    for ai, a in enumerate(assignments):
        for d in D:
            for s in S:
                if (ai, d, s) not in x:
                    continue
                cells = time_cells(a["classId"], s)
                for cell in cells:
                    teacher_slot.setdefault((a["teacherId"], d, cell), []).append(x[(ai, d, s)])
                    if a.get("isSplit") and a.get("splitTeacherId"):
                        teacher_slot.setdefault((a["splitTeacherId"], d, cell), []).append(x[(ai, d, s)])
    for key, lst in teacher_slot.items():
        if len(lst) > 1:
            m.Add(sum(lst) <= 1)

    # ===== XONA BIR VAQTDA BITTA DARS ===== (u ham real vaqt bo'yicha)
    room_slot = {}
    for ai, a in enumerate(assignments):
        for d in D:
            for s in S:
                if (ai, d, s) not in x:
                    continue
                cells = time_cells(a["classId"], s)
                for cell in cells:
                    if a.get("roomId"):
                        room_slot.setdefault((a["roomId"], d, cell), []).append(x[(ai, d, s)])
                    if a.get("isSplit") and a.get("splitRoomId"):
                        room_slot.setdefault((a["splitRoomId"], d, cell), []).append(x[(ai, d, s)])
    for key, lst in room_slot.items():
        if len(lst) > 1:
            m.Add(sum(lst) <= 1)

    # Bir kunda bir xil fan ko'pi bilan 1 marta (per class+subject+day)
    # ISTISNO: agar fanning haftalik soati kunlar sonidan ko'p bo'lsa (masalan Ingliz
    # 8 soat, 6 kun), "kunda 1 marta" bilan sig'maydi -> bunday fanga kunda 2 marta
    # (juft dars) ruxsat beramiz. Bu maktabda odatiy holat.
    subj_week_hours = {}
    for a in assignments:
        key = (a["classId"], a["subjectId"])
        subj_week_hours[key] = subj_week_hours.get(key, 0) + int(a.get("hoursPerWeek") or 0)
    # JUFT DARS (subject.doubleLesson = True): kuniga ko'pi bilan 2 dars va ular
    # ALBATTA KETMA-KET bo'lishi SHART (masalan 3- va 4-soat).
    csd = {}   # (cid, subid, d) -> {slot: [var, ...]}
    for ai, a in enumerate(assignments):
        for d in D:
            key = (a["classId"], a["subjectId"], d)
            for s in S:
                if (ai, d, s) in x:
                    csd.setdefault(key, {}).setdefault(s, []).append(x[(ai, d, s)])

    # yakka (juftlanmagan) darslar — objectivega yumshoq jarima sifatida qo'shiladi
    double_singles = []

    for key, slot_map in csd.items():
        cid, subid, d = key
        su = subjects.get(subid) or {}
        is_double = bool(su.get("doubleLesson"))
        all_vars = [v for lst in slot_map.values() for v in lst]
        if not all_vars:
            continue

        if is_double:
            # kuniga ko'pi bilan 2 dars
            if len(all_vars) > 2:
                m.Add(sum(all_vars) <= 2)

            # z[s] = shu fan shu sinfda (d, s) da bormi
            z = {}
            for s in S:
                zs = m.NewBoolVar(f"z_{cid}_{subid}_{d}_{s}")
                lst = slot_map.get(s, [])
                if lst:
                    m.Add(zs == sum(lst))
                else:
                    m.Add(zs == 0)
                z[s] = zs

            # QAT'IY: ikki dars orasida bo'shliq bo'lishi mumkin emas.
            # |s1 - s2| >= 2 bo'lgan har qanday juftlik taqiqlanadi.
            for s1 in S:
                for s2 in range(s1 + 2, slots):
                    m.Add(z[s1] + z[s2] <= 1)

            # YUMSHOQ: yakka dars uchun jarima -> solver juftlashga intiladi.
            # cnt in {0,1,2}; is2 = (cnt == 2); single = cnt - 2*is2 in {0,1}
            cnt = m.NewIntVar(0, 2, f"dcnt_{cid}_{subid}_{d}")
            m.Add(cnt == sum(z[s] for s in S))
            is2 = m.NewBoolVar(f"dis2_{cid}_{subid}_{d}")
            m.Add(cnt >= 2).OnlyEnforceIf(is2)
            m.Add(cnt <= 1).OnlyEnforceIf(is2.Not())
            double_singles.append(cnt - 2 * is2)
        else:
            wh = subj_week_hours.get((cid, subid), 0)
            max_per_day = 2 if wh > days else 1
            if len(all_vars) > max_per_day:
                m.Add(sum(all_vars) <= max_per_day)

    # O'qituvchi haftalik maksimal soati
    teacher_all = {}
    for ai, a in enumerate(assignments):
        for d in D:
            for s in S:
                if (ai, d, s) not in x:
                    continue
                teacher_all.setdefault(a["teacherId"], []).append(x[(ai, d, s)])
                if a.get("isSplit") and a.get("splitTeacherId"):
                    teacher_all.setdefault(a["splitTeacherId"], []).append(x[(ai, d, s)])
    for tid, lst in teacher_all.items():
        mx = teachers.get(tid, {}).get("maxHours")
        if mx:
            m.Add(sum(lst) <= int(mx))

    # =====================================================================
    #  MAQSAD FUNKSIYASI (objective)
    #  Ustuvorlik (leksikografik vaznlar bilan):
    #    W_PLACED   — imkon qadar ko'p dars joylashtirish (ENG MUHIM)
    #    W_CGAP     — sinf jadvalidagi bo'sh oynalarni minimallashtirish (kun o'rtasida)
    #    W_IMBAL    — kunlar orasidagi nomutanosiblikni minimallashtirish
    #    W_TGAP     — o'qituvchi jadvalidagi gaplarni minimallashtirish
    #    W_EARLY    — og'ir fanlarni ertaroq soatlarga (eng kichik ta'sir)
    #  MUHIM: bo'sh oyna endi QAT'IY man etilmagan — u YUMSHOQ maqsad. Shunda
    #  avval HAMMA dars joylashadi, keyin oynalar iloji boricha kamaytiriladi.
    #  (Qat'iy man etilsa, ko'p o'qituvchili maktabda darslar joyga sig'may qolardi.)
    # =====================================================================
    W_PLACED = 1000
    W_CGAP = 200    # sinf oynasi — juda kuchli jarima (darslar zich, oynasiz)
    W_IMBAL = 12
    W_TGAP = 2
    W_EARLY = 1
    W_PAIR = 60     # juft darsning yakka qolishiga jarima (kuchli, lekin yumshoq)

    obj = []

    # 1) Joylashtirilgan darslar soni (maksimal) — soatlar biriktirishga to'liq mos kelsin
    for ai in range(len(assignments)):
        p, hours = placed[ai]
        obj.append(W_PLACED * p)

    # 1b) SINF BO'SH OYNALARI (minimal): kun o'rtasidagi bo'sh slot.
    #     gap = (band emas) VA (o'sha kuni keyin dars bor). Faqat kun oxiridagi
    #     bo'sh joylar jarimasiz qoladi — aynan kerakli xatti-harakat.
    for c in class_ids:
        for d in D:
            for s in range(slots):
                after = m.NewBoolVar(f"cafter_{c}_{d}_{s}")
                laters = [y[(c, d, s2)] for s2 in range(s + 1, slots)]
                if laters:
                    m.AddMaxEquality(after, laters)
                else:
                    m.Add(after == 0)
                cgap = m.NewBoolVar(f"cgap_{c}_{d}_{s}")
                m.Add(cgap <= after)
                m.Add(cgap <= 1 - y[(c, d, s)])
                m.Add(cgap >= after - y[(c, d, s)])
                obj.append(-W_CGAP * cgap)

    # 1c) JUFT DARSLAR: yakka qolgan darsga jarima -> imkon qadar juft joylashadi
    for expr in double_singles:
        obj.append(-W_PAIR * expr)

    # 2) KUNLAR NOMUTANOSIBLIGI (minimal) + BARCHA ISH KUNLARIDAN FOYDALANISH.
    #    Har sinf uchun (eng band kun - eng bo'sh kun) ni kamaytiramiz, LEKIN faqat
    #    ish kunlari bo'yicha (band kunlar hisobga olinmaydi). Ayni paytda har ish
    #    kunida dars bo'lishini rag'batlantiramiz (bir kun butunlay bo'sh qolmasin).
    W_EMPTYDAY = 90   # ish kunini bo'sh qoldirish uchun jarima (kuchli)
    for c in class_ids:
        total = sum(int(a.get("hoursPerWeek") or 0)
                    for a in assignments if a["classId"] == c)
        if total <= 0:
            continue
        busy = class_busy.get(c, set())
        work_D = [d for d in D if d not in busy]
        loads = []
        for d in work_D:
            load = m.NewIntVar(0, slots, f"load_{c}_{d}")
            m.Add(load == sum(y[(c, d, s)] for s in S))
            loads.append(load)
            # bu ish kunida dars bormi? bo'lmasa jarima
            hasd = m.NewBoolVar(f"wd_{c}_{d}")
            m.Add(load >= 1).OnlyEnforceIf(hasd)
            m.Add(load == 0).OnlyEnforceIf(hasd.Not())
            # faqat yetarli darsli sinflarda bo'sh kunga jarima (kam darsli sinf bo'sh kun qoldirishi normal)
            if total >= len(work_D):
                obj.append(W_EMPTYDAY * hasd)
        if loads:
            mxl = m.NewIntVar(0, slots, f"mxl_{c}")
            mnl = m.NewIntVar(0, slots, f"mnl_{c}")
            m.AddMaxEquality(mxl, loads)
            m.AddMinEquality(mnl, loads)
            obj.append(-W_IMBAL * (mxl - mnl))

    # 3) O'QITUVCHI GAPLARI (minimal): o'qituvchining bir kunidagi darslar orasidagi
    #    bo'sh soatlar. Gap = (oxirgi dars slot - birinchi dars slot + 1) - darslar soni.
    #    DIQQAT: bu qism ko'p o'zgaruvchi qo'shadi. Katta masalada (ko'p sinf/o'qituvchi)
    #    uni o'chiramiz — shunda solver maksimal dars joylashtirishga e'tibor beradi
    #    va tez ishlaydi. Kichik masalada esa o'qituvchi jadvalini ixchamlaymiz.
    total_hours = sum(int(a.get("hoursPerWeek") or 0) for a in assignments)
    enable_tgap = (total_hours <= 120)   # ~15 sinfgacha to'liq optimallashtiramiz
    if enable_tgap:
        teacher_ids = [t["id"] for t in data["teachers"]]
        occ = {}
        for tid in teacher_ids:
            for d in D:
                for s in S:
                    lst = teacher_slot.get((tid, d, s), [])
                    ov = m.NewBoolVar(f"occ_{tid}_{d}_{s}")
                    if lst:
                        m.Add(ov == sum(lst))
                    else:
                        m.Add(ov == 0)
                    occ[(tid, d, s)] = ov
        for tid in teacher_ids:
            for d in D:
                day_occ = [occ[(tid, d, s)] for s in S]
                cnt = m.NewIntVar(0, slots, f"cnt_{tid}_{d}")
                m.Add(cnt == sum(day_occ))
                has_any = m.NewBoolVar(f"has_{tid}_{d}")
                m.Add(cnt >= 1).OnlyEnforceIf(has_any)
                m.Add(cnt == 0).OnlyEnforceIf(has_any.Not())
                first = m.NewIntVar(0, slots - 1, f"first_{tid}_{d}")
                last = m.NewIntVar(0, slots - 1, f"last_{tid}_{d}")
                for s in S:
                    m.Add(first <= s).OnlyEnforceIf(occ[(tid, d, s)])
                    m.Add(last >= s).OnlyEnforceIf(occ[(tid, d, s)])
                span = m.NewIntVar(0, slots, f"span_{tid}_{d}")
                m.Add(span == last - first + 1).OnlyEnforceIf(has_any)
                m.Add(span == 0).OnlyEnforceIf(has_any.Not())
                gaps_td = m.NewIntVar(0, slots, f"gaps_{tid}_{d}")
                m.Add(gaps_td == span - cnt)
                obj.append(-W_TGAP * gaps_td)

    # 4) og'ir fanlar ertaroq soatlarga (eng kichik ta'sir)
    for ai, a in enumerate(assignments):
        diff = subjects.get(a["subjectId"], {}).get("difficulty", 5)
        if diff >= 7:
            for d in D:
                for s in S:
                    if (ai, d, s) in x:
                        obj.append(W_EARLY * (slots - s) * x[(ai, d, s)])

    # ===== GREEDY WARM-START + BITTA SOLVE =====
    # Kuchsiz serverda (Render free, 0.1 CPU) CP-SAT'ga vaqt yetmasligi mumkin.
    # Yechim: avval Python'da TEZ greedy jadval tuzamiz (barcha qattiq shartlarga rioya
    # qilib), uni CP-SAT'ga "hint" (boshlang'ich yechim) sifatida beramiz. CP-SAT uni
    # incumbent qilib oladi — natija hech qachon greedy'dan yomon bo'lmaydi, vaqt
    # yetsa esa undan ancha yaxshi bo'ladi.
    placed_sum = sum(placed[ai][0] for ai in range(len(assignments)))

    seed = _greedy_seed(data, days, slots, teachers, subjects, assignments)

    m.Maximize(sum(obj))
    for key, var in x.items():
        m.AddHint(var, 1 if key in seed else 0)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 4
    solver.parameters.max_time_in_seconds = float(max_seconds)
    # Qat'iy urug' — takrorlanuvchanlikni oshiradi. TO'LIQ determinizm faqat
    # num_search_workers=1 da bo'ladi, lekin u juda sekin (o'lchandi: 15 soniyada
    # yechim topolmadi), shuning uchun 4 oqim qoldirildi. Natija urinishlar
    # orasida biroz farq qilishi mumkin — jadvalni saqlab qo'yish yoki
    # "Sinflarni qulflash" funksiyasi bundan himoya qiladi.
    solver.parameters.random_seed = 12345
    status = solver.Solve(m)

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }.get(status, str(status))

    entries = []
    unfilled = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for ai, a in enumerate(assignments):
            got = 0
            for d in D:
                for s in S:
                    if (ai, d, s) in x and solver.Value(x[(ai, d, s)]) == 1:
                        got += 1
                        entries.append({
                            "classId": a["classId"],
                            "subjectId": a["subjectId"],
                            "teacherId": a["teacherId"],
                            "roomId": a.get("roomId"),
                            "day": d, "lesson": s,
                            "group": "A" if a.get("isSplit") else None,
                        })
                        if a.get("isSplit") and a.get("splitTeacherId"):
                            entries.append({
                                "classId": a["classId"],
                                "subjectId": a["subjectId"],
                                "teacherId": a["splitTeacherId"],
                                "roomId": a.get("splitRoomId"),
                                "day": d, "lesson": s,
                                "group": "B",
                            })
            hours = int(a.get("hoursPerWeek") or 0)
            if got < hours:
                # `hours` maydoni ESKI KLIENT bilan moslik uchun saqlanadi.
                # Qolganlari — nega joylashmaganini tushuntirish uchun.
                unfilled.append({
                    "classId": a["classId"], "subjectId": a["subjectId"],
                    "teacherId": a["teacherId"], "hours": hours - got,
                    "requestedHours": hours,
                    "placedHours": got,
                    "missingHours": hours - got,
                    "reason": _unfilled_reason(a, data),
                })

    # ===== 2-BOSQICH: COMPACTION (zichlash) =====
    # CP-SAT natijasidan keyin darslarni kun ichida oldinga suramiz (bo'sh oynani
    # yo'qotish uchun) — FAQAT agar bu o'qituvchi/xona/fan cheklovlariga zid kelmasa.
    # Bu deterministik va tez; birinchi bosqichni buzmaydi.
    if entries:
        entries = _compact(entries, data, days, slots, teachers, subjects)

    # Qulflangan sinf darslarini natijaga qaytaramiz (ular o'zgarmagan)
    if fixed_entries:
        entries = entries + [dict(e) for e in fixed_entries]

    return {
        "entries": entries,
        "unfilled": unfilled,
        "status": status_name,
        "engineVersion": ENGINE_VERSION,
        "entry": "solve_timetable",
        "stats": {
            "objective": solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
            "wall_time": solver.WallTime(),
            "placed": len(entries),
        },
    }




def solve_safe(data, max_seconds=20):
    """XAVFSIZ KIRISH NUQTASI — API shuni chaqirishi kerak.

    Ikki muammoni hal qiladi:
      1) solve_timetable() INFEASIBLE qaytarsa entries BO'SH bo'lardi va
         foydalanuvchi hech narsa olmasdi.
      2) Kunlik yuk chegarasi qattiq bo'lsa taqsimot tekis, lekin ba'zi darslar
         joylashmay qolishi mumkin; chegara bo'sh bo'lsa hammasi joylashadi-yu
         taqsimot buziladi (bir kun 7 dars, boshqasi 3).

    Yechim — bosqichma-bosqich yumshatib, ENG YAXSHI natijani tanlash:
      0) ideal tekis taqsimot (floor..ceil)
      1) kunlik chegaraga ±1 zaxira
      2) kunlik chegara yo'q
      3) maxHours cheklovisiz (oxirgi chora)
    Har bosqichda to'liq joylashsa — darhol qaytariladi (keyingi bosqichlar
    faqat taqsimotni yomonlashtiradi). Aks holda barcha bosqichlar ichidan
    eng ko'p dars joylashgani, teng bo'lsa eng tekis taqsimlangani tanlanadi.
    """
    import copy

    total_needed = sum(int(a.get("hoursPerWeek") or 0)
                       for a in data.get("assignments", []))

    days_n = int(data["school"]["daysPerWeek"])
    # Sinfning haftalik soati: kunlar soniga yetadigan sinflarda bo'sh kun
    # bo'lmasligi kerak (yetmaydiganlarda bo'sh kun tabiiy).
    _cls_total = {}
    for a in data.get("assignments", []):
        _cls_total[a["classId"]] = _cls_total.get(a["classId"], 0) + int(a.get("hoursPerWeek") or 0)

    def day_counts(res):
        cnt = {}
        for e in (res.get("entries") or []):
            arr = cnt.setdefault(e["classId"], [0] * days_n)
            if 0 <= e["day"] < days_n:
                arr[e["day"]] += 1
        return cnt

    def imbalance(res):
        """Kunlar bo'yicha o'rtacha farq (kichik bo'lgani yaxshi)."""
        cnt = day_counts(res)
        if not cnt:
            return 999
        return sum(max(a) - min(a) for a in cnt.values()) / len(cnt)

    def avoidable_empty(res):
        """Oldini olsa bo'ladigan bo'sh kunlar soni.

        Sinfda haftalik soat kunlar soniga yetsa-yu, kun bo'sh qolsa — bu
        jadvalni maktab uchun yaroqsiz qiladi (bolalar haftada 2 kun keladi).
        Shunday jadval 13 ta qo'lda joylashtiriladigan darsdan ham yomonroq,
        shuning uchun tanlovda og'ir jarima bilan hisobga olinadi.
        """
        n = 0
        for cid, arr in day_counts(res).items():
            if _cls_total.get(cid, 0) >= days_n:
                n += sum(1 for v in arr if v == 0)
        return n

    def usable(res):
        return bool(res.get("entries")) and res.get("status") not in ("INFEASIBLE", "MODEL_INVALID")

    LEVELS = [
        (0, None),
        (1, "Kunlik dars soni chegarasi ±1 ga kengaytirildi — "
            "taqsimot biroz notekis bo'lishi mumkin."),
        (2, "Kunlik dars soni chegarasi olib tashlandi — "
            "kunlar bo'yicha farq sezilarli bo'lishi mumkin."),
        (3, "Bo'sh ish kuni taqiqi ham olib tashlandi — "
            "ba'zi sinflarda kun butunlay bo'sh qolishi mumkin."),
    ]

    best, best_key, warnings = None, None, []
    for lvl, note in LEVELS:
        # 0-darajaga (ideal tekis taqsimot) TO'LIQ vaqt beriladi, yumshatilgan
        # darajalarga yarmi. Sababi o'lchandi: vaqt kam bo'lsa 0-daraja hamma
        # darsni joylashtira olmay qoladi va tanlov yomonroq muvozanatli
        # darajaga o'tib ketardi (kunlar farqi 1 o'rniga 3 bo'lib qolardi).
        lvl_secs = max_seconds if lvl == 0 else max(8, max_seconds // 2)
        res = solve_timetable(data, max_seconds=lvl_secs, relax_days=lvl)
        if usable(res):
            placed = len(res.get("entries") or [])
            # Tanlov mezoni: bo'sh kun og'ir jarimalanadi (har biri ~3 darsga teng),
            # keyin joylashgan darslar soni, oxirida kunlar muvozanati.
            key = (placed - 3 * avoidable_empty(res), -imbalance(res))
            if best_key is None or key > best_key:
                best, best_key = res, key
                best["relaxed"] = lvl
                best["warnings"] = list(warnings) + ([note] if note else [])
            # To'liq joylashdi VA bo'sh kun yo'q — yumshatishning hojati yo'q
            if placed >= total_needed and avoidable_empty(res) == 0:
                best["engineVersion"] = ENGINE_VERSION
                best["entry"] = "solve_safe"
                return _finalize(best, data, total_needed)
        if note:
            warnings.append(note)

    if best is not None:
        best["engineVersion"] = ENGINE_VERSION
        best["entry"] = "solve_safe"
        return _finalize(best, data, total_needed)

    # Oxirgi chora: o'qituvchi haftalik limitini olib tashlash.
    # DIQQAT: bu AVTOMATIK bajarilmaydi. maxHours — real cheklov (mehnat
    # shartnomasi, stavka), uni jimgina buzish jadvalni yaroqsiz qiladi.
    # Faqat foydalanuvchi ataylab `allowTeacherOverload: true` qo'ysa ishlaydi.
    if not data.get("allowTeacherOverload"):
        res = solve_timetable(data, max_seconds=max_seconds, relax_days=3)
        res["relaxed"] = 3
        res["warnings"] = warnings + [
            "Barcha darslarni joylashtirib bo'lmadi. O'qituvchilarning haftalik "
            "maksimal soati (maxHours) cheklovi SAQLANDI — u avtomatik "
            "buzilmaydi. Joylashmagan darslar sabablari bilan `unfilled` "
            "ro'yxatida qaytarildi."]
        res["engineVersion"] = ENGINE_VERSION
        res["entry"] = "solve_safe"
        return _finalize(res, data, total_needed)

    warnings.append("O'qituvchilarning haftalik maksimal soati cheklovi olib tashlandi "
                    "(allowTeacherOverload=true) — yuklamani qo'lda tekshiring.")
    d2 = copy.deepcopy(data)
    for t in d2.get("teachers", []):
        t["maxHours"] = None
    res = solve_timetable(d2, max_seconds=max_seconds, relax_days=3)
    res["relaxed"] = 4
    res["warnings"] = warnings
    res["engineVersion"] = ENGINE_VERSION
    res["entry"] = "solve_safe"
    if not (res.get("entries") or []):
        res["warnings"].append(
            "Jadval tuzilmadi. Sabab odatda ma'lumotda: sinf haftalik soati kunlik "
            "sig'imdan ortiq, yoki o'qituvchilarda metodik kun/band soatlar juda ko'p.")
    return res







def solve_variants(data, max_seconds=30, count=3):
    """BIR NECHTA JADVAL VARIANTI — foydalanuvchi eng mosini tanlaydi.

    Sabab: "hech qanday bo'sh oyna bo'lmasin" va "hamma dars joylashsin"
    talablari ba'zan bir-biriga zid. Qattiq tekis taqsimot bir-ikki darsni
    joylashtirmay qoldirishi mumkin; chegarani bo'shatsak hammasi joylashadi-yu
    kunlar notekis bo'ladi. Bitta "to'g'ri" javob yo'q — maktab o'zi tanlashi
    kerak. Shuning uchun uch xil ustuvorlik bilan yechamiz.

    Har variant validatordan o'tadi; qat'iy cheklov buzilgani chiqarilmaydi.
    Qaytadi: {"variants": [...], "best": 0, va eng yaxshisining maydonlari}
    """
    profiles = [
        (0, "Muvozanatli", "Darslar hafta kunlariga eng tekis taqsimlangan"),
        (1, "Moslashuvchan", "Kunlik chegara ±1 — ko'proq dars joylashadi"),
        (2, "To'liq", "Chegarasiz — maksimal dars joylashtiriladi"),
    ]
    profiles = profiles[:max(1, min(3, count))]
    total_needed = sum(int(a.get("hoursPerWeek") or 0)
                       for a in data.get("assignments", []))

    per = max(8, int(max_seconds / max(1, len(profiles))))
    out_list = []
    seen = set()
    for lvl, label, note in profiles:
        try:
            r = solve_timetable(data, max_seconds=per, relax_days=lvl)
        except Exception:                                     # noqa: BLE001
            continue
        if not (r.get("entries") or []):
            continue
        r = _finalize(r, data, total_needed)
        if not r.get("validation", {}).get("valid"):
            continue                       # nuqsonli variant ko'rsatilmaydi
        st = r.get("stats", {})
        sig = (st.get("placed"), st.get("classGaps"), st.get("dailyImbalance"))
        if sig in seen:
            continue                       # bir xil natijani ikki marta ko'rsatmaymiz
        seen.add(sig)
        out_list.append({
            "label": label,
            "note": note,
            "relaxed": lvl,
            "entries": r["entries"],
            "unfilled": r.get("unfilled") or [],
            "stats": st,
            "validation": r.get("validation"),
            "status": r.get("status"),
        })

    if not out_list:
        # Birorta ham toza variant chiqmadi — odatdagi xavfsiz yo'lga o'tamiz
        res = solve_safe(data, max_seconds=max_seconds)
        res["variants"] = []
        return res

    def score(v):
        st = v["stats"]
        # Ustuvorlik: joylashgan dars > sinf oynasi > muvozanat > o'qituvchi oynasi
        return (st.get("placed", 0) * 1000
                - st.get("classGaps", 0) * 200
                - float(st.get("dailyImbalance", 0)) * 30
                - st.get("teacherGaps", 0))

    out_list.sort(key=score, reverse=True)

    # DOMINATSIYA FILTRI: agar variant boshqasidan HAMMA ko'rsatkichda
    # yomon bo'lsa, uni ko'rsatmaymiz. Foydalanuvchiga aniq yomon variantni
    # tanlash imkonini berish — yordam emas, chalkashlik.
    def dominated(v, other):
        a, b = v["stats"], other["stats"]
        return (b.get("placed", 0) >= a.get("placed", 0)
                and b.get("classGaps", 0) <= a.get("classGaps", 0)
                and float(b.get("dailyImbalance", 0)) <= float(a.get("dailyImbalance", 0))
                and b.get("teacherGaps", 0) <= a.get("teacherGaps", 0)
                and (b.get("placed", 0) > a.get("placed", 0)
                     or b.get("classGaps", 0) < a.get("classGaps", 0)
                     or float(b.get("dailyImbalance", 0)) < float(a.get("dailyImbalance", 0))
                     or b.get("teacherGaps", 0) < a.get("teacherGaps", 0)))

    kept = [v for i, v in enumerate(out_list)
            if not any(dominated(v, o) for j, o in enumerate(out_list) if i != j)]
    if kept:
        out_list = kept
    best = out_list[0]
    res = {
        "entries": best["entries"],
        "unfilled": best["unfilled"],
        "status": best["status"],
        "stats": best["stats"],
        "validation": best["validation"],
        "relaxed": best["relaxed"],
        "engineVersion": ENGINE_VERSION,
        "entry": "solve_variants",
        "variants": [{"label": v["label"], "note": v["note"],
                      "entries": v["entries"], "unfilled": v["unfilled"],
                      "stats": v["stats"]} for v in out_list],
        "warnings": [],
    }
    return res


def _fixed_slot(subject):
    """Fanning qat'iy belgilangan (kun, soat) joyi yoki None.

    Klientdagi fixedSlotOf() bilan bir xil shakl: {"day": 0, "lesson": 0}.
    Masalan "Kelajak soati" — dushanba (0) birinchi soat (0).
    """
    if not subject:
        return None
    f = subject.get("fixedSlot")
    if not f:
        return None
    d, l = f.get("day"), f.get("lesson")
    if d is None or l is None:
        return None
    try:
        return (int(d), int(l))
    except (TypeError, ValueError):
        return None


def _finalize(res, data, total_needed):
    """Javobni yakunlaydi: validatordan o'tkazadi va stats to'ldiradi.

    Bu yerda jadval O'ZGARTIRILMAYDI — faqat tekshiriladi va o'lchanadi.
    Post-processing bosqichlari (compaction) allaqachon tugagan, shuning uchun
    validator ularning natijasini ham qamrab oladi.
    """
    entries = res.get("entries") or []
    v = validate_schedule(entries, data)
    placed = sum(1 for e in entries if e.get("group") != "B")
    st = res.get("stats") or {}
    st.update({
        "requested": total_needed,
        "placed": placed,
        "unfilled": sum(int(u.get("missingHours") or u.get("hours") or 0)
                        for u in (res.get("unfilled") or [])),
        "classGaps": v["stats"]["classGaps"],
        "teacherGaps": v["stats"]["teacherGaps"],
        "dailyImbalance": v["stats"]["dailyImbalance"],
    })
    res["stats"] = st
    res["validation"] = {"valid": v["valid"], "errors": v["errors"],
                         "warnings": v["warnings"][:20]}
    if not v["valid"]:
        # Qat'iy cheklov buzilgan bo'lsa buni YASHIRMAYMIZ — klient
        # verifySolverOutput() orqali javobni rad etib, zaxiraga o'tadi.
        res["status"] = "INVALID"
        res.setdefault("warnings", []).append(
            "Jadval yakuniy tekshiruvdan o'tmadi: " + "; ".join(v["errors"][:3]))
    return res


def validate_schedule(entries, data):
    """YAKUNIY NAZORATCHI — jadval qat'iy cheklovlarni buzmaganini tekshiradi.

    Solverdan ALOHIDA yozilgan: solver o'zini o'zi "to'g'ri" deb e'lon qila
    olmasligi kerak. Har qanday post-processing (compaction, ta'mirlash)
    dan KEYIN chaqiriladi.

    Qaytaradi: {"valid": bool, "errors": [...], "warnings": [...], "stats": {...}}
    valid=False bo'lsa jadval frontendga yuborilmasligi kerak.
    """
    errors, warnings = [], []
    school = data.get("school", {})
    days = _num(school.get("daysPerWeek"), 6)

    teachers = {t["id"]: t for t in data.get("teachers", [])}
    classes = {c["id"]: c for c in data.get("classes", [])}
    subjects = {sb["id"]: sb for sb in data.get("subjects", [])}

    def cshift(cid):
        c = classes.get(cid)
        return _num(c.get("shift"), 1) if c else 1

    def cname(cid):
        c = classes.get(cid)
        return c.get("name", cid) if c else cid

    def tname(tid):
        t = teachers.get(tid)
        return t.get("name", tid) if t else tid

    # --- To'qnashuvlar: REAL VAQT bo'yicha (smena raqami emas) ---
    # Bir kunda har juftlikni tekshiramiz, lekin faqat kerakli guruhlar ichida —
    # O(n^2) butun jadval bo'yicha emas, o'qituvchi/xona bo'yicha.
    by_teacher, by_room, by_class = {}, {}, {}
    for e in entries:
        by_teacher.setdefault((e.get("teacherId"), e.get("day")), []).append(e)
        if e.get("roomId"):
            by_room.setdefault((e["roomId"], e.get("day")), []).append(e)
        by_class.setdefault((e.get("classId"), e.get("day")), []).append(e)

    def time_clash(e1, e2):
        return slots_overlap(school, cshift(e1["classId"]), e1["lesson"],
                             cshift(e2["classId"]), e2["lesson"])

    for (tid, d), lst in by_teacher.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                # Bo'linadigan dars: A va B guruh BIR sinfda, bir slotda —
                # ular turli o'qituvchida bo'ladi, shuning uchun bu yerga tushmaydi.
                if a["classId"] == b["classId"] and a["lesson"] == b["lesson"]:
                    continue
                if time_clash(a, b):
                    errors.append("O'qituvchi to'qnashuvi: %s, %d-kun, %s va %s"
                                  % (tname(tid), d + 1, cname(a["classId"]), cname(b["classId"])))

    for (rid, d), lst in by_room.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                if a["classId"] == b["classId"] and a["lesson"] == b["lesson"]:
                    continue
                if time_clash(a, b):
                    errors.append("Xona to'qnashuvi: %s, %d-kun, %d-soat"
                                  % (rid, d + 1, a["lesson"] + 1))

    for (cid, d), lst in by_class.items():
        seen = {}
        for e in lst:
            key = (e["lesson"], e.get("group"))
            if key in seen:
                errors.append("Sinf to'qnashuvi: %s, %d-kun, %d-soat"
                              % (cname(cid), d + 1, e["lesson"] + 1))
            seen[key] = 1

    # --- O'qituvchi bandligi, metodik kun, fan cheklovi, smena chegarasi ---
    for e in entries:
        t = teachers.get(e.get("teacherId"))
        if t:
            if t.get("methodicalDay") is not None and int(t["methodicalDay"]) == e["day"]:
                errors.append("Metodik kun buzildi: %s, %d-kun" % (tname(e["teacherId"]), e["day"] + 1))
            if (t.get("unavailable") or {}).get("%d-%d" % (e["day"], e["lesson"])):
                errors.append("O'qituvchi band deb belgilangan vaqt: %s, %d-kun %d-soat"
                              % (tname(e["teacherId"]), e["day"] + 1, e["lesson"] + 1))
        sub = subjects.get(e.get("subjectId"))
        fx = _fixed_slot(sub)
        if fx is not None and (e["day"], e["lesson"]) != fx:
            errors.append("Belgilangan vaqt buzildi: %s %d-kun %d-soatda bo'lishi kerak"
                          % (sub.get("name", ""), fx[0] + 1, fx[1] + 1))
        if sub and (sub.get("unavailable") or {}).get("%d-%d" % (e["day"], e["lesson"])):
            errors.append("Fan cheklovi buzildi: %s, %d-kun %d-soat"
                          % (sub.get("name", ""), e["day"] + 1, e["lesson"] + 1))
        c = classes.get(e.get("classId"))
        if c and (c.get("busyDays") or {}).get(str(e["day"])) or (c and (c.get("busyDays") or {}).get(e["day"])):
            errors.append("Sinf dam kunida dars: %s, %d-kun" % (cname(e["classId"]), e["day"] + 1))
        per = shift_cfg(school, cshift(e.get("classId")))["lessons"]
        if e["lesson"] < 0 or e["lesson"] >= per:
            errors.append("Smena chegarasidan tashqari: %s, %d-soat"
                          % (cname(e["classId"]), e["lesson"] + 1))
        if e["day"] < 0 or e["day"] >= days:
            errors.append("Ish kunidan tashqari: %s, %d-kun" % (cname(e["classId"]), e["day"] + 1))

    # --- Sinf kunidagi PREFIKS qoidasi (bo'sh oyna bo'lmasin) ---
    class_gaps = 0
    for (cid, d), lst in by_class.items():
        used = set(e["lesson"] for e in lst)
        if not used:
            continue
        for sslot in range(max(used)):
            if sslot not in used:
                class_gaps += 1
                warnings.append("Bo'sh oyna: %s, %d-kun, %d-soat"
                                % (cname(cid), d + 1, sslot + 1))

    # --- Haftalik soat: rejadan ORTIQ bo'lmasin (kam bo'lishi unfilled orqali) ---
    need, got = {}, {}
    for a in data.get("assignments", []):
        k = (a.get("classId"), a.get("subjectId"))
        need[k] = need.get(k, 0) + int(a.get("hoursPerWeek") or 0)
    for e in entries:
        if e.get("group") == "B":
            continue
        k = (e.get("classId"), e.get("subjectId"))
        got[k] = got.get(k, 0) + 1
    for k, v in got.items():
        if v > need.get(k, 0):
            errors.append("Rejadan ortiqcha soat: %s, %d o'rniga %d"
                          % (cname(k[0]), need.get(k, 0), v))

    # --- O'qituvchi haftalik limiti ---
    tload = {}
    for e in entries:
        tload[e.get("teacherId")] = tload.get(e.get("teacherId"), 0) + 1
    for tid, n in tload.items():
        t = teachers.get(tid)
        if not t:
            continue
        mx = _num(t.get("maxHours"), 0)
        if mx and n > mx:
            errors.append("Haftalik limit buzildi: %s, %d soat (maksimum %d)"
                          % (tname(tid), n, mx))

    # --- O'qituvchi bo'sh oynalari (faqat ogohlantirish) ---
    teacher_gaps = 0
    for (tid, d), lst in by_teacher.items():
        used = set(e["lesson"] for e in lst)
        if used:
            teacher_gaps += sum(1 for x in range(max(used)) if x not in used)

    # --- Kunlik nomutanosiblik ---
    per_class_day = {}
    for e in entries:
        if e.get("group") == "B":
            continue
        per_class_day.setdefault(e["classId"], [0] * days)
        if 0 <= e["day"] < days:
            per_class_day[e["classId"]][e["day"]] += 1
    imbalance = 0
    if per_class_day:
        imbalance = sum(max(v) - min(v) for v in per_class_day.values()) / len(per_class_day)

    return {
        "valid": len(errors) == 0,
        "errors": errors[:50],
        "warnings": warnings[:50],
        "stats": {
            "classGaps": class_gaps,
            "teacherGaps": teacher_gaps,
            "dailyImbalance": round(imbalance, 2),
            "errorCount": len(errors),
        },
    }


def _unfilled_reason(a, data):
    """Nega bu dars joylashmadi — eng ehtimoliy sababni matn qilib qaytaradi.

    Diagnostika `diagnose()` bilan bir xil mantiqda, lekin bitta biriktirish
    uchun. Frontend buni to'g'ridan-to'g'ri foydalanuvchiga ko'rsatadi.
    """
    school = data.get("school", {})
    days = _num(school.get("daysPerWeek"), 6)
    need = int(a.get("hoursPerWeek") or 0)

    teachers = {t["id"]: t for t in data.get("teachers", [])}
    classes = {c["id"]: c for c in data.get("classes", [])}
    subjects = {sb["id"]: sb for sb in data.get("subjects", [])}

    t = teachers.get(a.get("teacherId"))
    c = classes.get(a.get("classId"))
    sub = subjects.get(a.get("subjectId"))

    if t is None:
        return "O'qituvchi topilmadi (o'chirilgan yoki tanlanmagan)"
    if c is None:
        return "Sinf topilmadi (o'chirilgan)"

    per = shift_cfg(school, _num(c.get("shift"), 1))["lessons"]

    # 1) O'qituvchining haftalik limiti
    t_need = 0
    for x in data.get("assignments", []):
        h = int(x.get("hoursPerWeek") or 0)
        if x.get("teacherId") == t["id"]:
            t_need += h
        if x.get("isSplit") and x.get("splitTeacherId") == t["id"]:
            t_need += h
    mx = _num(t.get("maxHours"), 0)
    if mx and t_need > mx:
        return ("O'qituvchi %s ga jami %d soat biriktirilgan, ruxsat etilgan "
                "maksimum %d soat" % (t.get("name", t["id"]), t_need, mx))

    # 2) O'qituvchining bo'sh kataklari yetadimi
    work_days = days - (1 if t.get("methodicalDay") is not None else 0)
    blocked = 0
    for k, v in (t.get("unavailable") or {}).items():
        if not v:
            continue
        try:
            dd = int(str(k).split("-")[0])
        except (ValueError, IndexError):
            continue
        if dd < days and str(t.get("methodicalDay")) != str(dd):
            blocked += 1
    cap = work_days * per - blocked
    if t_need > cap:
        return ("O'qituvchi %s ning bo'sh vaqti yetmaydi: %d soat kerak, "
                "%d ta bo'sh katak bor (metodik kun va band soatlar hisobga olindi)"
                % (t.get("name", t["id"]), t_need, cap))

    # 3) Sinf sig'imi
    c_need = sum(int(x.get("hoursPerWeek") or 0)
                 for x in data.get("assignments", [])
                 if x.get("classId") == c["id"])
    busy = sum(1 for v in (c.get("busyDays") or {}).values() if v)
    c_cap = (days - busy) * per
    if c_need > c_cap:
        return ("Sinf %s sig'imi yetmaydi: %d soat kerak, kunlik %d soatdan "
                "%d kunga jami %d katak" % (c.get("name", c["id"]), c_need,
                                            per, days - busy, c_cap))

    # 4) Fan cheklovi
    if sub is not None:
        blocked_s = sum(1 for k, v in (sub.get("unavailable") or {}).items()
                        if v and str(k).split("-")[0].isdigit()
                        and int(str(k).split("-")[0]) < days)
        if blocked_s >= days * per:
            return "Fan %s uchun barcha kun/soatlar taqiqlangan" % sub.get("name", "")
        if blocked_s > days * per * 0.5:
            return ("Fan %s uchun kun/soat cheklovlari juda qattiq (%d katak taqiqlangan)"
                    % (sub.get("name", ""), blocked_s))

    # 5) Maxsus xona
    if a.get("roomId"):
        r_need = sum(int(x.get("hoursPerWeek") or 0)
                     for x in data.get("assignments", [])
                     if x.get("roomId") == a["roomId"])
        two = int(school.get("shiftMode") or 1) == 2
        r_cap = days * shift_cfg(school, 1)["lessons"]
        if two:
            r_cap += days * shift_cfg(school, 2)["lessons"]
        if r_need > r_cap:
            return "Maxsus xona band: %d soat talab, sig'imi %d soat" % (r_need, r_cap)

    return ("Sinf va o'qituvchining bo'sh vaqtlari mos kelmadi "
            "(boshqa darslar bilan to'qnashuv)")


def _compact(entries, data, days, slots, teachers, subjects):
    """Darslarni oldinga suradi (cheklovlarni buzmasdan). Split (guruhli) darslar
    juft yozuv (A/B) bo'lgani uchun ular YAXLIT hujayra sifatida birga ko'chadi.

    IKKI SMENA: bandlik slot raqami emas, REAL VAQT bo'yicha hisoblanadi.
    Aks holda 1-smenaning oxirgi soati bilan 2-smenaning birinchi soati
    ustma-ust tushganda o'qituvchi uchun to'qnashuv hosil bo'lishi mumkin edi."""
    school = data["school"]
    class_shift = {c["id"]: int(c.get("shift") or 1) for c in data["classes"]}
    shift_slots = {sh: shift_cfg(school, sh)["lessons"] for sh in (1, 2)}
    _cells_cache = {}

    def time_cells(cid, s):
        """Shu (sinf, soat) egallaydigan vaqt oynalari — ikkala smena bo'yicha."""
        sh = class_shift.get(cid, 1)
        key = (sh, s)
        if key in _cells_cache:
            return _cells_cache[key]
        cells = [(sh, s)]
        other = 2 if sh == 1 else 1
        for j in range(shift_slots.get(other, 0)):
            if slots_overlap(school, sh, s, other, j):
                cells.append((other, j))
        _cells_cache[key] = cells
        return cells

    def cls_slots(cid):
        return shift_slots.get(class_shift.get(cid, 1), slots)

    teacher_busy = set()
    room_busy = set()
    for e in entries:
        for cell in time_cells(e["classId"], e["lesson"]):
            teacher_busy.add((e["teacherId"], e["day"], cell))
            if e.get("roomId"):
                room_busy.add((e["roomId"], e["day"], cell))

    def teacher_ok(tid, d, s):
        t = teachers.get(tid)
        if not t:
            return True
        if t.get("methodicalDay") is not None and int(t["methodicalDay"]) == d:
            return False
        if t.get("unavailable", {}).get(f"{d}-{s}"):
            return False
        return True

    def subject_ok(subid, d, s):
        su = subjects.get(subid)
        return not (su and su.get("unavailable", {}).get(f"{d}-{s}"))

    def cells_of(cid, d):
        """(cid,d) kunidagi hujayralar: lesson -> [entry, ...] (split juftlik birga)."""
        out = {}
        for e in entries:
            if e["classId"] == cid and e["day"] == d:
                out.setdefault(e["lesson"], []).append(e)
        return out

    def can_place_cell(cell, d, s):
        for e in cell:
            if s >= cls_slots(e["classId"]):        # bu smenada bunday soat yo'q
                return False
            for c2 in time_cells(e["classId"], s):
                if (e["teacherId"], d, c2) in teacher_busy:
                    return False
                if e.get("roomId") and (e["roomId"], d, c2) in room_busy:
                    return False
            if not teacher_ok(e["teacherId"], d, s):
                return False
            if not subject_ok(e["subjectId"], d, s):
                return False
        return True

    def move_cell(cell, nd, ns):
        for e in cell:
            for c2 in time_cells(e["classId"], e["lesson"]):
                teacher_busy.discard((e["teacherId"], e["day"], c2))
                if e.get("roomId"):
                    room_busy.discard((e["roomId"], e["day"], c2))
        for e in cell:
            e["day"] = nd
            e["lesson"] = ns
            for c2 in time_cells(e["classId"], ns):
                teacher_busy.add((e["teacherId"], nd, c2))
                if e.get("roomId"):
                    room_busy.add((e["roomId"], nd, c2))

    class_ids = sorted(set(e["classId"] for e in entries))

    # 1-BOSQICH: kun ichida oldinga surish
    changed = True
    guard = 0
    while changed and guard < 60:
        changed = False
        guard += 1
        for cid in class_ids:
            for d in range(days):
                occ = cells_of(cid, d)
                for target in range(slots):
                    if target in occ:
                        continue
                    nxt = None
                    for s in range(target + 1, slots):
                        if s in occ:
                            nxt = s
                            break
                    if nxt is None:
                        break
                    cell = occ[nxt]
                    # o'z joyidan bo'shatib tekshiramiz (o'zi bilan to'qnashmasin)
                    for e in cell:
                        # TUZATILDI: time_cells() allaqachon (smena, soat) kortejlarini
                        # qaytaradi — uni yana time_cells() ga uzatish TypeError berardi
                        # va _compact butunlay ishlamay qolardi (ayniqsa 2 smenali maktabda).
                        for c2 in time_cells(e["classId"], nxt):
                            teacher_busy.discard((e["teacherId"], d, c2))
                            if e.get("roomId"):
                                room_busy.discard((e["roomId"], d, c2))
                    ok = can_place_cell(cell, d, target)
                    for e in cell:
                        for c2 in time_cells(e["classId"], nxt):
                            teacher_busy.add((e["teacherId"], d, c2))
                            if e.get("roomId"):
                                room_busy.add((e["roomId"], d, c2))
                    if ok:
                        move_cell(cell, d, target)
                        occ = cells_of(cid, d)
                        changed = True

    # 2-BOSQICH: kunlararo — bo'sh oynani boshqa kunning OXIRGI darsi bilan to'ldirish
    changed = True
    guard = 0
    while changed and guard < 60:
        changed = False
        guard += 1
        for cid in class_ids:
            for d in range(days):
                occ = cells_of(cid, d)
                if not occ:
                    continue
                last = max(occ)
                gap_slot = None
                for s in range(last):
                    if s not in occ:
                        gap_slot = s
                        break
                if gap_slot is None:
                    continue
                subj_today = set()
                for cell in occ.values():
                    for e in cell:
                        subj_today.add(e["subjectId"])
                moved = False
                for od in range(days):
                    if od == d or moved:
                        continue
                    oocc = cells_of(cid, od)
                    if not oocc:
                        continue
                    o_last = max(oocc)
                    cell = oocc[o_last]
                    if any(e["subjectId"] in subj_today for e in cell):
                        continue
                    for e in cell:
                        for _c in time_cells(e["classId"], o_last):
                            teacher_busy.discard((e["teacherId"], od, _c))
                        if e.get("roomId"):
                            for _c in time_cells(e["classId"], o_last):
                                room_busy.discard((e["roomId"], od, _c))
                    ok = can_place_cell(cell, d, gap_slot)
                    if ok:
                        for e in cell:
                            e["day"] = d
                            e["lesson"] = gap_slot
                            for _c in time_cells(e["classId"], gap_slot):
                                teacher_busy.add((e["teacherId"], d, _c))
                            if e.get("roomId"):
                                for _c in time_cells(e["classId"], gap_slot):
                                    room_busy.add((e["roomId"], d, _c))
                        moved = True
                        changed = True
                    else:
                        for e in cell:
                            for _c in time_cells(e["classId"], o_last):
                                teacher_busy.add((e["teacherId"], od, _c))
                            if e.get("roomId"):
                                for _c in time_cells(e["classId"], o_last):
                                    room_busy.add((e["roomId"], od, _c))

    # 3-BOSQICH: agressiv oyna to'ldirish — bo'sh oynaga boshqa kundagi ISTALGAN
    # darsni (nafaqat oxirgisini) ko'chiramiz, agar u dars o'z kunida oxirgi bo'lsa
    # (ya'ni uni olib ketish yangi oyna ochmasa) yoki ko'chirilgach o'z kuni zichlansa.
    changed = True
    guard = 0
    while changed and guard < 60:
        changed = False
        guard += 1
        for cid in class_ids:
            for d in range(days):
                occ = cells_of(cid, d)
                if not occ:
                    continue
                last = max(occ)
                gap_slot = None
                for s in range(last):
                    if s not in occ:
                        gap_slot = s
                        break
                if gap_slot is None:
                    continue
                subj_today = set()
                for cell in occ.values():
                    for e in cell:
                        subj_today.add(e["subjectId"])
                moved = False
                for od in range(days):
                    if od == d or moved:
                        continue
                    oocc = cells_of(cid, od)
                    if not oocc:
                        continue
                    o_last = max(oocc)
                    # o'sha kundagi ISTALGAN slotdagi darsni sinaymiz
                    for oslot in sorted(oocc.keys(), reverse=True):
                        if moved:
                            break
                        cell = oocc[oslot]
                        if any(e["subjectId"] in subj_today for e in cell):
                            continue
                        for e in cell:
                            for _c in time_cells(e["classId"], oslot):
                                teacher_busy.discard((e["teacherId"], od, _c))
                            if e.get("roomId"):
                                for _c in time_cells(e["classId"], oslot):
                                    room_busy.discard((e["roomId"], od, _c))
                        ok = can_place_cell(cell, d, gap_slot)
                        if ok:
                            for e in cell:
                                e["day"] = d
                                e["lesson"] = gap_slot
                                for _c in time_cells(e["classId"], gap_slot):
                                    teacher_busy.add((e["teacherId"], d, _c))
                                if e.get("roomId"):
                                    for _c in time_cells(e["classId"], gap_slot):
                                        room_busy.add((e["roomId"], d, _c))
                            moved = True
                            changed = True
                        else:
                            for e in cell:
                                for _c in time_cells(e["classId"], oslot):
                                    teacher_busy.add((e["teacherId"], od, _c))
                                if e.get("roomId"):
                                    for _c in time_cells(e["classId"], oslot):
                                        room_busy.add((e["roomId"], od, _c))
                # oslotdan bo'shagan joyni to'ldirish keyingi 1-bosqich takrorida hal bo'ladi

    # 4-BOSQICH: qolgan oynalarni yana bir marta kun ichida zichlash
    changed = True
    guard = 0
    while changed and guard < 60:
        changed = False
        guard += 1
        for cid in class_ids:
            for d in range(days):
                occ = cells_of(cid, d)
                for target in range(slots):
                    if target in occ:
                        continue
                    nxt = None
                    for s in range(target + 1, slots):
                        if s in occ:
                            nxt = s
                            break
                    if nxt is None:
                        break
                    cell = occ[nxt]
                    for e in cell:
                        # TUZATILDI: time_cells() allaqachon (smena, soat) kortejlarini
                        # qaytaradi — uni yana time_cells() ga uzatish TypeError berardi
                        # va _compact butunlay ishlamay qolardi (ayniqsa 2 smenali maktabda).
                        for c2 in time_cells(e["classId"], nxt):
                            teacher_busy.discard((e["teacherId"], d, c2))
                            if e.get("roomId"):
                                room_busy.discard((e["roomId"], d, c2))
                    ok = can_place_cell(cell, d, target)
                    for e in cell:
                        for c2 in time_cells(e["classId"], nxt):
                            teacher_busy.add((e["teacherId"], d, c2))
                            if e.get("roomId"):
                                room_busy.add((e["roomId"], d, c2))
                    if ok:
                        move_cell(cell, d, target)
                        occ = cells_of(cid, d)
                        changed = True

    return entries


if __name__ == "__main__":
    import json, sys
    data = json.load(sys.stdin)
    # MUHIM: solve_timetable emas, solve_safe chaqiriladi — INFEASIBLE holatida
    # bo'sh jadval o'rniga yumshatilgan yechim qaytadi.
    print(json.dumps(solve_safe(data)))
