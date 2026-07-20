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
  "stats": {...}
}

Cheklovlar:
  HARD (buzilmaydi):
   - sinf bir vaqtda bitta dars
   - o'qituvchi bir vaqtda bitta dars (parallel yo'q), split ikkinchi o'qituvchi ham
   - metodik kun / o'qituvchi bandligi / fan kun-soat cheklovi
   - xona bir vaqtda bitta dars
   - bir kunda bir xil fan ko'pi bilan 1 marta (per class)
   - o'qituvchi haftalik maksimal soati
   - BO'SH OYNA YO'Q: har sinf har kunda darslar 0-slotdan ketma-ket (prefix)
  SOFT (optimallashtiriladi):
   - imkon qadar ko'p dars joylashtirilsin (placed maksimal)
   - kunlar muvozanati (bir kun juda ko'p / juda kam bo'lmasin)
   - og'ir fanlar ertaroq soatlarga
"""
from ortools.sat.python import cp_model
import math


def _expand_half_assignments(assignments):
    """Kasr (0.5) soatli biriktirishlarni butun + yarim qismlarga ajratadi.
    Yarim (0.5) dars 1 ta jismoniy slot egallaydi (bir hafta o'tiladi/o'tilmaydi),
    lekin u '_half=True' bilan belgilanadi -> model uni sinf kunining OXIRGI
    darsiga qo'yadi. Masalan: 1.5 soat -> 1 butun + 1 yarim; 0.5 -> 1 yarim."""
    out = []
    for a in assignments:
        hrs = float(a.get("hoursPerWeek") or 0)
        whole = int(math.floor(hrs + 1e-9))
        has_half = (hrs - whole) >= 0.49
        if whole > 0:
            b = dict(a); b["hoursPerWeek"] = whole; b["_half"] = False
            out.append(b)
        if has_half:
            c = dict(a); c["hoursPerWeek"] = 1; c["_half"] = True
            out.append(c)
    return out


def diagnose(data):
    """Jadval to'liq chiqmasligining ANIQ sabablarini topadi (o'zbekcha xabarlar).
    Bu — matematik tekshiruv: hech qanday dastur bu holatlarni joylashtira olmaydi."""
    school = data["school"]
    days = int(school["daysPerWeek"])
    slots = int(school["lessonsPerDay"])
    total_slots = days * slots
    teachers = {t["id"]: t for t in data["teachers"]}
    msgs = []

    # 1) SINF: haftalik slotdan ko'p dars biriktirilganmi?
    cls_names = {c["id"]: c.get("name", c["id"]) for c in data["classes"]}
    cls_load = {}
    for a in data["assignments"]:
        cls_load[a["classId"]] = cls_load.get(a["classId"], 0) + math.ceil(float(a.get("hoursPerWeek") or 0))
    for cid, load in cls_load.items():
        if load > total_slots:
            msgs.append(
                f"SINF {cls_names.get(cid, cid)}: {load} soat biriktirilgan, lekin haftada "
                f"{total_slots} ta joy bor ({days} kun × {slots} soat) — {load - total_slots} soati "
                f"hech qachon sig'maydi. Yechim: bu sinf soatlarini kamaytiring."
            )

    # 2) O'QITUVCHI: bo'sh vaqti / max soatidan ko'p yuk berilganmi?
    t_load = {}
    for a in data["assignments"]:
        h = math.ceil(float(a.get("hoursPerWeek") or 0))
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
    """Tez greedy jadval — CP-SAT uchun boshlang'ich yechim (hint).

    MUHIM: natija model'ning QAT'IY cheklovlariga mos bo'lishi SHART, aks holda
    CP-SAT hint'ni rad etadi va noldan qidiradi (natija yomonlashadi).
    Shuning uchun:
      * darslar har sinf-kunida 1-soatdan KETMA-KET joylashadi (bo'sh oyna yo'q),
      * kunlik yuqori chegara (hi_day) hurmat qilinadi,
      * kunlar imkon qadar tekis to'ldiriladi (min_day ni qanoatlantirish uchun).
    """
    import math

    t_busy = set()      # (tid, d, s)
    r_busy = set()      # (rid, d, s)
    csd = set()         # (cid, subid, d) — bir kunda bir fan
    t_hours = {}        # tid -> jami soat
    c_dayload = {}      # (cid, d) -> shu kundagi darslar soni (= keyingi bo'sh slot)

    class_busy = {}
    for c in data.get("classes", []):
        bd = c.get("busyDays") or {}
        class_busy[c["id"]] = set(int(k) for k, v in bd.items() if v)

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

    # Har sinf uchun kunlik yuqori chegara (model bilan bir xil mantiq)
    MIN_PER_DAY = 4
    hi_of = {}
    for c in data.get("classes", []):
        cid = c["id"]
        total_c = sum(int(a.get("hoursPerWeek") or 0)
                      for a in assignments if a["classId"] == cid)
        work_days = max(1, days - len(class_busy.get(cid, set())))
        min_day = 1 if total_c < MIN_PER_DAY * work_days else MIN_PER_DAY
        even = math.ceil(total_c / work_days)
        slack = 2 if slots - even >= 2 else (1 if slots - even >= 1 else 0)
        hi_of[cid] = min(slots, max(min_day, even + slack))

    # Eng cheklangan (og'ir yuklangan) o'qituvchilar birinchi joylashsin
    tasks = []
    for ai, a in enumerate(assignments):
        for _ in range(int(a.get("hoursPerWeek") or 0)):
            tasks.append((ai, a))

    load_cache = {}
    for a in assignments:
        tid = a["teacherId"]
        load_cache[tid] = load_cache.get(tid, 0) + int(a.get("hoursPerWeek") or 0)

    def constraint_score(item):
        ai, a = item
        t = teachers.get(a["teacherId"], {})
        load = load_cache.get(a["teacherId"], 0)
        meth = 1 if t.get("methodicalDay") is not None else 0
        unav = len([1 for v in (t.get("unavailable") or {}).values() if v])
        return -(load * 2 + meth * slots + unav)

    tasks.sort(key=constraint_score)

    seed = set()
    unplaced = []

    def try_place(ai, a):
        """Faqat KETMA-KET slotga (s == kundagi darslar soni) joylashtiradi."""
        cid, subid, tid = a["classId"], a["subjectId"], a["teacherId"]
        stid = a.get("splitTeacherId") if a.get("isSplit") else None
        busy_days = class_busy.get(cid, set())
        hi = hi_of.get(cid, slots)
        best, best_score = None, None
        for d in range(days):
            if d in busy_days:
                continue
            if (cid, subid, d) in csd:
                continue
            s = c_dayload.get((cid, d), 0)      # <-- ketma-ketlik kafolati
            if s >= slots or s >= hi:
                continue
            if (tid, d, s) in t_busy or not teacher_ok(tid, d, s):
                continue
            if stid and ((stid, d, s) in t_busy or not teacher_ok(stid, d, s)):
                continue
            if not subject_ok(subid, d, s):
                continue
            if a.get("roomId") and (a["roomId"], d, s) in r_busy:
                continue
            # Kunlarni tekis to'ldiramiz: kam yuklangan kun afzal
            sc = -s * 10 - c_dayload.get((cid, d), 0) * 3
            if best_score is None or sc > best_score:
                best_score, best = sc, (d, s)
        if best is None:
            return False
        if not teacher_cap_ok(tid) or (stid and not teacher_cap_ok(stid)):
            return False
        d, s = best
        seed.add((ai, d, s))
        t_busy.add((tid, d, s))
        if stid:
            t_busy.add((stid, d, s))
        if a.get("roomId"):
            r_busy.add((a["roomId"], d, s))
        csd.add((cid, subid, d))
        t_hours[tid] = t_hours.get(tid, 0) + 1
        if stid:
            t_hours[stid] = t_hours.get(stid, 0) + 1
        c_dayload[(cid, d)] = s + 1
        return True

    for ai, a in enumerate_tasks(tasks):
        if not try_place(ai, a):
            unplaced.append((ai, a))

    # 2-urinish: qolganlarini yana bir marta sinaymiz (holat o'zgargan bo'lishi mumkin)
    if unplaced:
        again, unplaced = unplaced, []
        for ai, a in again:
            if not try_place(ai, a):
                unplaced.append((ai, a))

    return seed


def enumerate_tasks(tasks):
    """(ai, a) juftliklarini qaytaradi (tasks allaqachon (ai,a) formatida)."""
    for ai, a in tasks:
        yield ai, a


def solve_timetable(data, max_seconds=20):
    school = data["school"]
    days = int(school["daysPerWeek"])
    slots = int(school["lessonsPerDay"])
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

    # Kasr (0.5) soatli darslarni butun + yarim qismlarga ajratamiz.
    # Shundan keyin barcha hoursPerWeek — butun son (int() endi hech narsa yo'qotmaydi).
    assignments = _expand_half_assignments(assignments)

    m = cp_model.CpModel()

    D = range(days)
    S = range(slots)

    # qulflangan darslar egallagan (o'qituvchi/xona) slotlar — bandlik
    fixed_teacher = set()
    fixed_room = set()
    for e in fixed_entries:
        fixed_teacher.add((e["teacherId"], e["day"], e["lesson"]))
        if e.get("roomId"):
            fixed_room.add((e["roomId"], e["day"], e["lesson"]))

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
                # qulflangan darslar bilan to'qnashuv bo'lmasin
                if ok and (a["teacherId"], d, s) in fixed_teacher:
                    ok = False
                if ok and a.get("isSplit") and a.get("splitTeacherId") and (a["splitTeacherId"], d, s) in fixed_teacher:
                    ok = False
                if ok and a.get("roomId") and (a["roomId"], d, s) in fixed_room:
                    ok = False
                # sinfning band kuni — dars qo'yilmaydi
                if ok and d in class_busy.get(a["classId"], set()):
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
        for d in D:
            for s in range(slots - 1):
                m.Add(y[(c, d, s)] >= y[(c, d, s + 1)])

    # ===== QAT'IY: YARIM (0.5) DARS — sinf kunining ENG OXIRGI darsi bo'lsin =====
    # 0.5 soatlik dars bir hafta o'tiladi, bir hafta o'tilmaydi. O'tilmagan haftada
    # o'rtada bo'sh oyna qolmasligi uchun uni har doim kunning oxirgi darsiga qo'yamiz.
    # Ya'ni: agar yarim dars (d, s) da bo'lsa, o'sha sinfda (d, s+1) da dars bo'lmasin.
    for ai, a in enumerate(assignments):
        if not a.get("_half"):
            continue
        cid = a["classId"]
        for d in D:
            for s in range(slots - 1):
                if (ai, d, s) in x:
                    m.Add(x[(ai, d, s)] + y[(cid, d, s + 1)] <= 1)

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
        # ===== YUQORI CHEGARA (hi_day) =====
        # MUHIM TUZATISH: ilgari hi_day = ceil(total/work_days) QAT'IY edi.
        # Bu katta maktablarda halokatli: masalan 30 soat / 6 kun -> hi_day=5,
        # ya'ni 6- va 7-soatlar BUTUNLAY bloklanardi. Natijada bo'sh kataklar
        # ko'rinib tursa ham darslar joylashmasdi (o'qituvchilar erta soatlarda
        # to'qnashib qolardi).
        #
        # Endi: qat'iy chegaraga BO'SHLIQ (slack) beramiz — solver zarur bo'lganda
        # kunni uzaytira oladi. Tekis taqsimot esa ob'ektiv funksiyadagi W_IMBAL
        # jarimasi orqali (YUMSHOQ) ta'minlanadi — ya'ni imkon bo'lsa tekis,
        # imkon bo'lmasa dars joylashadi.
        even = math.ceil(total_c / work_days) if work_days else slots
        slack = 2 if slots - even >= 2 else (1 if slots - even >= 1 else 0)
        hi_day = min(slots, max(min_day, even + slack))
        # Agar darslar soni ish kunlariga yetsa (har kunga kamida 1 tadan), HAR ish
        # kunida kamida 1 dars bo'lishini QAT'IY talab qilamiz -> bir kun butunlay
        # bo'sh qolmaydi (masalan Dushanba bo'sh qolmaydi).
        force_all_days = total_c >= work_days
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
                # har ish kunida kamida forced_min dars (qat'iy) -> bo'sh kun yo'q
                m.Add(day_load >= forced_min)
                m.Add(day_load <= hi_day)
            else:
                # kam darsli sinf: kun bo'sh bo'lishi mumkin, lekin dars bo'lsa >=min_day
                has_day = m.NewBoolVar(f"hasday_{c}_{d}")
                m.Add(day_load >= 1).OnlyEnforceIf(has_day)
                m.Add(day_load == 0).OnlyEnforceIf(has_day.Not())
                m.Add(day_load >= min_day).OnlyEnforceIf(has_day)
                m.Add(day_load <= hi_day)

    # O'qituvchi bir vaqtda bitta dars (parallel yo'q) — split teacher ham
    teacher_slot = {}  # (tid,d,s) -> list of vars
    for ai, a in enumerate(assignments):
        for d in D:
            for s in S:
                if (ai, d, s) not in x:
                    continue
                teacher_slot.setdefault((a["teacherId"], d, s), []).append(x[(ai, d, s)])
                if a.get("isSplit") and a.get("splitTeacherId"):
                    teacher_slot.setdefault((a["splitTeacherId"], d, s), []).append(x[(ai, d, s)])
    for key, lst in teacher_slot.items():
        if len(lst) > 1:
            m.Add(sum(lst) <= 1)

    # Xona bir vaqtda bitta dars (asosiy va split xona)
    room_slot = {}
    for ai, a in enumerate(assignments):
        for d in D:
            for s in S:
                if (ai, d, s) not in x:
                    continue
                if a.get("roomId"):
                    room_slot.setdefault((a["roomId"], d, s), []).append(x[(ai, d, s)])
                if a.get("isSplit") and a.get("splitRoomId"):
                    room_slot.setdefault((a["splitRoomId"], d, s), []).append(x[(ai, d, s)])
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
    csd = {}
    for ai, a in enumerate(assignments):
        for d in D:
            key = (a["classId"], a["subjectId"], d)
            for s in S:
                if (ai, d, s) in x:
                    csd.setdefault(key, []).append(x[(ai, d, s)])
    for key, lst in csd.items():
        cid, subid, d = key
        wh = subj_week_hours.get((cid, subid), 0)
        max_per_day = 2 if wh > days else 1
        if len(lst) > max_per_day:
            m.Add(sum(lst) <= max_per_day)

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
    used_fallback = False
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        source = {key for key, var in x.items() if solver.Value(var) == 1}
    else:
        # ===== ZAXIRA: CP-SAT vaqt ichida yechim topmadi (UNKNOWN/INFEASIBLE) =====
        # Bunday holatda AVVAL bo'sh jadval qaytarilardi — foydalanuvchi hech narsa
        # ko'rmasdi. Endi greedy yechimдан foydalanamiz: u barcha qat'iy qoidalarga
        # mos (oynasiz, to'qnashuvsiz), shuning uchun xavfsiz.
        source = set(seed)
        used_fallback = True

    got_count = {}
    for (ai, d, s) in sorted(source):
        a = assignments[ai]
        got_count[ai] = got_count.get(ai, 0) + 1
        entries.append({
            "classId": a["classId"],
            "subjectId": a["subjectId"],
            "teacherId": a["teacherId"],
            "roomId": a.get("roomId"),
            "day": d, "lesson": s,
            "group": "A" if a.get("isSplit") else None,
            "isHalf": bool(a.get("_half")),
        })
        if a.get("isSplit") and a.get("splitTeacherId"):
            entries.append({
                "classId": a["classId"],
                "subjectId": a["subjectId"],
                "teacherId": a["splitTeacherId"],
                "roomId": a.get("splitRoomId"),
                "day": d, "lesson": s,
                "group": "B",
                "isHalf": bool(a.get("_half")),
            })
    for ai, a in enumerate(assignments):
        hours = int(a.get("hoursPerWeek") or 0)
        got = got_count.get(ai, 0)
        if got < hours:
            rem = hours - got
            # yarim dars joylashmasa — frontend uni ½ chip sifatida ko'rsatishi uchun 0.5
            unfilled.append({
                "classId": a["classId"], "subjectId": a["subjectId"],
                "teacherId": a["teacherId"],
                "hours": (rem * 0.5) if a.get("_half") else rem,
            })

    # ===== 2-BOSQICH: COMPACTION (zichlash) =====
    # CP-SAT natijasidan keyin darslarni kun ichida oldinga suramiz (bo'sh oynani
    # yo'qotish uchun) — FAQAT agar bu o'qituvchi/xona/fan cheklovlariga zid kelmasa.
    # Bu deterministik va tez; birinchi bosqichni buzmaydi.
    if entries:
        entries = _compact(entries, data, days, slots, teachers, subjects)

    # ===== 3-BOSQICH: QOLGAN DARSLARNI JOYLASHTIRISH (repair) =====
    # CP-SAT vaqt yetmasligi sababli ba'zi darslarni qoldirishi mumkin, holbuki
    # jadvalda bo'sh kataklar bor. Bu bosqich shunday darslarni deterministik
    # ravishda bo'sh joylarga qo'yadi — barcha qat'iy qoidalarga rioya qilgan holda:
    #   * sinf kunida darslar ketma-ket (bo'sh oyna paydo bo'lmaydi),
    #   * o'qituvchi/xona to'qnashuvi yo'q,
    #   * bir fan bir kunda takrorlanmaydi,
    #   * metodik kun va band kunlar hurmat qilinadi.
    if unfilled:
        entries, unfilled = _fill_remaining(
            entries, unfilled, data, days, slots, teachers, subjects,
            fixed_teacher, fixed_room
        )

    # Qulflangan sinf darslarini natijaga qaytaramiz (ular o'zgarmagan)
    if fixed_entries:
        entries = entries + [dict(e) for e in fixed_entries]

    return {
        "entries": entries,
        "unfilled": unfilled,
        "status": ("FALLBACK" if used_fallback else status_name),
        "stats": {
            "objective": solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
            "wall_time": solver.WallTime(),
            "placed": len(entries),
            "fallback": used_fallback,
            "solver_status": status_name,
        },
    }


def _compact(entries, data, days, slots, teachers, subjects):
    """Darslarni oldinga suradi (cheklovlarni buzmasdan). Split (guruhli) darslar
    juft yozuv (A/B) bo'lgani uchun ular YAXLIT hujayra sifatida birga ko'chadi."""
    teacher_busy = set()
    room_busy = set()
    for e in entries:
        teacher_busy.add((e["teacherId"], e["day"], e["lesson"]))
        if e.get("roomId"):
            room_busy.add((e["roomId"], e["day"], e["lesson"]))

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
            if (e["teacherId"], d, s) in teacher_busy:
                return False
            if e.get("roomId") and (e["roomId"], d, s) in room_busy:
                return False
            if not teacher_ok(e["teacherId"], d, s):
                return False
            if not subject_ok(e["subjectId"], d, s):
                return False
        return True

    def move_cell(cell, nd, ns):
        for e in cell:
            teacher_busy.discard((e["teacherId"], e["day"], e["lesson"]))
            if e.get("roomId"):
                room_busy.discard((e["roomId"], e["day"], e["lesson"]))
        for e in cell:
            e["day"] = nd
            e["lesson"] = ns
            teacher_busy.add((e["teacherId"], nd, ns))
            if e.get("roomId"):
                room_busy.add((e["roomId"], nd, ns))

    class_ids = sorted(set(e["classId"] for e in entries))

    # YARIM (0.5) dars — ko'chirilmaydi (u sinf kunining oxirgi darsi bo'lib qolishi
    # kerak). Shuningdek, tepasida yarim dars turgan oddiy dars ham ko'chirilmaydi
    # (aks holda yarim dars osilib, oxirgi bo'lmay qoladi).
    def _cell_has_half(cell):
        return any(e.get("isHalf") for e in cell)

    def _half_above(cid, d, slot):
        for e in entries:
            if e["classId"] == cid and e["day"] == d and e["lesson"] > slot and e.get("isHalf"):
                return True
        return False

    def _locked(cell, cid, d, slot):
        return _cell_has_half(cell) or _half_above(cid, d, slot)

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
                    if _locked(cell, cid, d, nxt):
                        continue
                    # o'z joyidan bo'shatib tekshiramiz (o'zi bilan to'qnashmasin)
                    for e in cell:
                        teacher_busy.discard((e["teacherId"], d, nxt))
                        if e.get("roomId"):
                            room_busy.discard((e["roomId"], d, nxt))
                    ok = can_place_cell(cell, d, target)
                    for e in cell:
                        teacher_busy.add((e["teacherId"], d, nxt))
                        if e.get("roomId"):
                            room_busy.add((e["roomId"], d, nxt))
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
                    if _locked(cell, cid, od, o_last):
                        continue
                    if any(e["subjectId"] in subj_today for e in cell):
                        continue
                    for e in cell:
                        teacher_busy.discard((e["teacherId"], od, o_last))
                        if e.get("roomId"):
                            room_busy.discard((e["roomId"], od, o_last))
                    ok = can_place_cell(cell, d, gap_slot)
                    if ok:
                        for e in cell:
                            e["day"] = d
                            e["lesson"] = gap_slot
                            teacher_busy.add((e["teacherId"], d, gap_slot))
                            if e.get("roomId"):
                                room_busy.add((e["roomId"], d, gap_slot))
                        moved = True
                        changed = True
                    else:
                        for e in cell:
                            teacher_busy.add((e["teacherId"], od, o_last))
                            if e.get("roomId"):
                                room_busy.add((e["roomId"], od, o_last))

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
                        if _locked(cell, cid, od, oslot):
                            continue
                        if any(e["subjectId"] in subj_today for e in cell):
                            continue
                        for e in cell:
                            teacher_busy.discard((e["teacherId"], od, oslot))
                            if e.get("roomId"):
                                room_busy.discard((e["roomId"], od, oslot))
                        ok = can_place_cell(cell, d, gap_slot)
                        if ok:
                            for e in cell:
                                e["day"] = d
                                e["lesson"] = gap_slot
                                teacher_busy.add((e["teacherId"], d, gap_slot))
                                if e.get("roomId"):
                                    room_busy.add((e["roomId"], d, gap_slot))
                            moved = True
                            changed = True
                        else:
                            for e in cell:
                                teacher_busy.add((e["teacherId"], od, oslot))
                                if e.get("roomId"):
                                    room_busy.add((e["roomId"], od, oslot))
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
                    if _locked(cell, cid, d, nxt):
                        continue
                    for e in cell:
                        teacher_busy.discard((e["teacherId"], d, nxt))
                        if e.get("roomId"):
                            room_busy.discard((e["roomId"], d, nxt))
                    ok = can_place_cell(cell, d, target)
                    for e in cell:
                        teacher_busy.add((e["teacherId"], d, nxt))
                        if e.get("roomId"):
                            room_busy.add((e["roomId"], d, nxt))
                    if ok:
                        move_cell(cell, d, target)
                        occ = cells_of(cid, d)
                        changed = True

    return entries


if __name__ == "__main__":
    import json, sys
    data = json.load(sys.stdin)
    print(json.dumps(solve_timetable(data)))


def _fill_remaining(entries, unfilled, data, days, slots, teachers, subjects,
                    fixed_teacher=None, fixed_room=None):
    """CP-SAT joylashtira olmagan darslarni bo'sh kataklarga qo'yadi.

    Qat'iy qoidalar (hech qachon buzilmaydi):
      * sinf bir vaqtda bitta dars,
      * o'qituvchi bir vaqtda bitta dars (qulflangan darslar ham hisobga olinadi),
      * xona bir vaqtda bitta dars,
      * bir fan bir kunda takrorlanmaydi,
      * metodik kun / band kun / band soat,
      * dars faqat kunning KEYINGI bo'sh slotiga qo'yiladi -> bo'sh oyna paydo bo'lmaydi,
      * yarim (0.5) dars faqat kunning oxirgi darsi bo'la oladi.

    Qaytaradi: (yangilangan entries, qolgan unfilled)
    """
    fixed_teacher = fixed_teacher or set()
    fixed_room = fixed_room or set()

    class_busy = {}
    for c in data.get("classes", []):
        bd = c.get("busyDays") or {}
        class_busy[c["id"]] = set(int(k) for k, v in bd.items() if v)

    # Joriy holat
    occupied = {}       # (cid, d, s) -> True
    dayload = {}        # (cid, d) -> nechta dars (ketma-ket bo'lgani uchun = keyingi slot)
    t_busy = set(fixed_teacher)
    r_busy = set(fixed_room)
    csd = set()         # (cid, subid, d)
    half_at = set()     # (cid, d, s) — yarim dars turgan joy

    for e in entries:
        key = (e["classId"], e["day"], e["lesson"])
        occupied[key] = True
        t_busy.add((e["teacherId"], e["day"], e["lesson"]))
        if e.get("roomId"):
            r_busy.add((e["roomId"], e["day"], e["lesson"]))
        csd.add((e["classId"], e["subjectId"], e["day"]))
        if e.get("isHalf"):
            half_at.add(key)

    # Har sinf-kun uchun band slotlar soni (ketma-ket deb hisoblaymiz)
    for (cid, d, s) in list(occupied.keys()):
        dayload[(cid, d)] = max(dayload.get((cid, d), 0), s + 1)

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

    # Assignment ma'lumotlarini topish uchun indeks (split/xona uchun)
    a_index = {}
    for a in data.get("assignments", []):
        a_index[(a["classId"], a["subjectId"], a["teacherId"])] = a

    still = []
    for u in unfilled:
        cid, subid, tid = u["classId"], u["subjectId"], u["teacherId"]
        hours = u.get("hours") or 0
        is_half = (abs(hours - round(hours)) > 1e-6)   # 0.5, 1.5 ...
        a = a_index.get((cid, subid, tid), {})
        stid = a.get("splitTeacherId") if a.get("isSplit") else None
        rid = a.get("roomId")
        srid = a.get("splitRoomId")

        remaining = hours
        step = 0.5 if is_half else 1
        guard = 0
        while remaining > 1e-6 and guard < 200:
            guard += 1
            placed_one = False
            # Kunlarni kam yuklangan tartibda ko'rib chiqamiz (tekis taqsimot)
            day_order = sorted(range(days), key=lambda dd: dayload.get((cid, dd), 0))
            for d in day_order:
                if d in class_busy.get(cid, set()):
                    continue
                if (cid, subid, d) in csd:
                    continue                      # bir fan kunda bir marta
                s = dayload.get((cid, d), 0)
                if s >= slots:
                    continue                      # kun to'lgan
                if (cid, d, s) in occupied:
                    continue
                # yarim dars ustidagi katakka dars qo'ymaymiz
                if (cid, d, s - 1) in half_at:
                    continue
                if not teacher_ok(tid, d, s) or (tid, d, s) in t_busy:
                    continue
                if stid and (not teacher_ok(stid, d, s) or (stid, d, s) in t_busy):
                    continue
                if not subject_ok(subid, d, s):
                    continue
                if rid and (rid, d, s) in r_busy:
                    continue
                if srid and (srid, d, s) in r_busy:
                    continue

                # JOYLASHTIRAMIZ
                half_flag = (step == 0.5)
                entries.append({
                    "classId": cid, "subjectId": subid, "teacherId": tid,
                    "roomId": rid, "day": d, "lesson": s,
                    "group": "A" if stid else None, "isHalf": half_flag,
                })
                if stid:
                    entries.append({
                        "classId": cid, "subjectId": subid, "teacherId": stid,
                        "roomId": srid, "day": d, "lesson": s,
                        "group": "B", "isHalf": half_flag,
                    })
                occupied[(cid, d, s)] = True
                dayload[(cid, d)] = s + 1
                t_busy.add((tid, d, s))
                if stid:
                    t_busy.add((stid, d, s))
                if rid:
                    r_busy.add((rid, d, s))
                if srid:
                    r_busy.add((srid, d, s))
                csd.add((cid, subid, d))
                if half_flag:
                    half_at.add((cid, d, s))
                remaining -= step
                placed_one = True
                break
            if not placed_one:
                break

        if remaining > 1e-6:
            still.append({"classId": cid, "subjectId": subid,
                          "teacherId": tid, "hours": round(remaining, 1)})

    return entries, still
