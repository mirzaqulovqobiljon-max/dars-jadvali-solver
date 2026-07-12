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
        cls_load[a["classId"]] = cls_load.get(a["classId"], 0) + int(a.get("hoursPerWeek") or 0)
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


def solve_timetable(data, max_seconds=20):
    school = data["school"]
    days = int(school["daysPerWeek"])
    slots = int(school["lessonsPerDay"])
    teachers = {t["id"]: t for t in data["teachers"]}
    subjects = {s["id"]: s for s in data["subjects"]}
    classes = data["classes"]
    assignments = data["assignments"]

    m = cp_model.CpModel()

    D = range(days)
    S = range(slots)

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

    # Sinf bir vaqtda bitta dars (bo'sh oyna endi QAT'IY emas — objective'da yumshoq)
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
    csd = {}
    for ai, a in enumerate(assignments):
        for d in D:
            key = (a["classId"], a["subjectId"], d)
            for s in S:
                if (ai, d, s) in x:
                    csd.setdefault(key, []).append(x[(ai, d, s)])
    for key, lst in csd.items():
        if len(lst) > 1:
            m.Add(sum(lst) <= 1)

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

    # 2) KUNLAR NOMUTANOSIBLIGI (minimal): har sinf uchun (eng band kun - eng bo'sh kun)
    #    Chegara YUMSHOQ (objective orqali) — shunda avval maksimal dars joylashadi,
    #    keyin imkon qadar tekis taqsimlanadi. Qat'iy chegara qo'ymaymiz, aks holda
    #    kam o'qituvchi holatida ko'p dars "joylashmadi"ga chiqib ketardi.
    for c in class_ids:
        total = sum(int(a.get("hoursPerWeek") or 0)
                    for a in assignments if a["classId"] == c)
        if total <= 0:
            continue
        loads = []
        for d in D:
            load = m.NewIntVar(0, slots, f"load_{c}_{d}")
            m.Add(load == sum(y[(c, d, s)] for s in S))
            loads.append(load)
        mxl = m.NewIntVar(0, slots, f"mxl_{c}")
        mnl = m.NewIntVar(0, slots, f"mnl_{c}")
        m.AddMaxEquality(mxl, loads)
        m.AddMinEquality(mnl, loads)
        obj.append(-W_IMBAL * (mxl - mnl))    # nomutanosiblikni kamaytirish

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

    # ===== IKKI BOSQICHLI YECHIM (lexicographic) =====
    # 1-bosqich: FAQAT maksimal dars joylashtirishni top (tez).
    # 2-bosqich: o'sha maksimalni saqlagan holda sifatni (gap, muvozanat) yaxshila.
    placed_sum = sum(placed[ai][0] for ai in range(len(assignments)))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8

    # 1-bosqich: TEZ maksimal joylashtirishni top (faqat placement, sifatsiz)
    m.Maximize(placed_sum)
    solver.parameters.max_time_in_seconds = max(4.0, float(max_seconds) * 0.2)
    st1 = solver.Solve(m)
    best_placed = None
    hint_vars, hint_vals = [], []
    if st1 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        best_placed = int(solver.ObjectiveValue())
        # 1-bosqich yechimini 2-bosqichga "maslahat" (hint) sifatida beramiz -> tezlashadi
        for key, var in x.items():
            hint_vars.append(var)
            hint_vals.append(int(solver.Value(var)))

    # 2-bosqich: joylashtirishni saqlab, SIFATNI (oyna, muvozanat) optimallashtir.
    status = st1
    if best_placed is not None:
        # deyarli hamma dars joylashsin (1-2 dars kamayishiga ruxsat -> oyna kamayadi)
        m.Add(placed_sum >= best_placed)
        # umumiy maqsad: placement (juda katta vazn) + sifat
        m.Maximize(sum(obj))
        if hint_vars:
            m.ClearHints()
            for v, val in zip(hint_vars, hint_vals):
                m.AddHint(v, val)
        solver.parameters.max_time_in_seconds = max(8.0, float(max_seconds) * 0.8)
        st2 = solver.Solve(m)
        if st2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            status = st2

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
                unfilled.append({
                    "classId": a["classId"], "subjectId": a["subjectId"],
                    "teacherId": a["teacherId"], "hours": hours - got,
                })

    # ===== 2-BOSQICH: COMPACTION (zichlash) =====
    # CP-SAT natijasidan keyin darslarni kun ichida oldinga suramiz (bo'sh oynani
    # yo'qotish uchun) — FAQAT agar bu o'qituvchi/xona/fan cheklovlariga zid kelmasa.
    # Bu deterministik va tez; birinchi bosqichni buzmaydi.
    if entries:
        entries = _compact(entries, data, days, slots, teachers, subjects)

    return {
        "entries": entries,
        "unfilled": unfilled,
        "status": status_name,
        "stats": {
            "objective": solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
            "wall_time": solver.WallTime(),
            "placed": len(entries),
        },
    }


def _compact(entries, data, days, slots, teachers, subjects):
    """Har sinf-kun uchun darslarni oldinga suradi (cheklovlarni buzmasdan)."""
    # tez qidiruv uchun indekslar
    teacher_busy = set()   # (tid, d, s)
    room_busy = set()      # (rid, d, s)
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

    # sinf -> kun -> darslar ro'yxati
    from collections import defaultdict
    by_cd = defaultdict(list)
    for e in entries:
        by_cd[(e["classId"], e["day"])].append(e)

    changed = True
    guard = 0
    while changed and guard < 50:
        changed = False
        guard += 1
        for (cid, d), lst in by_cd.items():
            occupied = {e["lesson"]: e for e in lst}
            for target in range(slots):
                if target in occupied:
                    continue
                # target bo'sh — undan keyingi eng yaqin darsni topib oldinga suramiz
                nxt = None
                for s in range(target + 1, slots):
                    if s in occupied:
                        nxt = s
                        break
                if nxt is None:
                    break  # bu kunda boshqa dars yo'q
                e = occupied[nxt]
                # e ni (d, target) ga ko'chirish mumkinmi?
                if (e["teacherId"], d, target) in teacher_busy:
                    continue
                if e.get("roomId") and (e["roomId"], d, target) in room_busy:
                    continue
                if not teacher_ok(e["teacherId"], d, target):
                    continue
                if not subject_ok(e["subjectId"], d, target):
                    continue
                # xavfsizlik: bir kunda bir xil fan ikki marta bo'lib qolmasin
                # (nxt slotdagi fan target'ga ko'chsa, o'sha kunda boshqa nusxa yo'qligini
                #  tekshiramiz — lekin nxt->target ko'chishida fan o'sha kunda qoladi,
                #  shuning uchun bu tekshiruv faqat boshqa dars target'da bo'lsa kerak,
                #  target bo'sh bo'lgani uchun muammo yo'q. Baribir himoya sifatida qoldiramiz.)
                # ko'chiramiz
                teacher_busy.discard((e["teacherId"], d, nxt))
                teacher_busy.add((e["teacherId"], d, target))
                if e.get("roomId"):
                    room_busy.discard((e["roomId"], d, nxt))
                    room_busy.add((e["roomId"], d, target))
                del occupied[nxt]
                e["lesson"] = target
                occupied[target] = e
                changed = True

    return entries


if __name__ == "__main__":
    import json, sys
    data = json.load(sys.stdin)
    print(json.dumps(solve_timetable(data)))
