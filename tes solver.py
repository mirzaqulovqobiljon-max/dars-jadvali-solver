"""Solver uchun majburiy testlar (1-10). Ishga tushirish: python3 test_solver.py"""
import importlib.util
import collections

spec = importlib.util.spec_from_file_location("sv", "solver.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print("  %-52s %s %s" % (name, "OK" if cond else "XATO", detail))


def school(shift_mode=1, days=6, lessons=6, **kw):
    d = {"name": "T", "shiftMode": shift_mode, "daysPerWeek": days,
         "lessonsPerDay": lessons, "lessonDuration": 45, "breakDuration": 10,
         "startShift1": "08:00", "startShift2": "14:00",
         "bigBreakAfter": 3, "bigBreakDuration": 20}
    d.update(kw)
    return d


def gaps_of(entries, data):
    by = collections.defaultdict(set)
    for e in entries:
        if e.get("group") == "B":
            continue
        by[(e["classId"], e["day"])].add(e["lesson"])
    return sum(1 for k, ls in by.items() for s in range(max(ls)) if s not in ls)


# ---------------------------------------------------------------- TEST 1
print("\nTEST 1: oddiy 1 smena — hamma dars joylashsin, oyna 0")
d = {"school": school(), "rooms": [],
     "teachers": [{"id": "t%d" % i, "name": "T%d" % i, "maxHours": 30,
                   "methodicalDay": None, "unavailable": {}} for i in range(4)],
     "subjects": [{"id": "s%d" % i, "name": "F%d" % i, "difficulty": 5,
                   "unavailable": {}} for i in range(4)],
     "classes": [{"id": "c%d" % i, "name": "%d-A" % (i + 1), "shift": 1,
                  "busyDays": {}} for i in range(3)],
     "assignments": []}
for ci in range(3):
    for si in range(4):
        d["assignments"].append({"id": "a%d_%d" % (ci, si), "teacherId": "t%d" % si,
                                 "subjectId": "s%d" % si, "classId": "c%d" % ci,
                                 "hoursPerWeek": 4, "roomId": None, "isSplit": False})
r = S.solve_safe(d, max_seconds=12)
tot = sum(a["hoursPerWeek"] for a in d["assignments"])
check("hamma dars joylashdi", len(r["entries"]) == tot, "%d/%d" % (len(r["entries"]), tot))
check("bo'sh oyna yo'q", gaps_of(r["entries"], d) == 0)
check("validator: valid", r["validation"]["valid"], str(r["validation"]["errors"][:1]))
check("stats to'ldirildi", r["stats"].get("requested") == tot and "classGaps" in r["stats"])

# ---------------------------------------------------------------- TEST 2
print("\nTEST 2: o'qituvchi to'qnashuvi bo'lmasin")
v = S.validate_schedule(r["entries"], d)
check("o'qituvchi to'qnashuvi yo'q",
      not any("O'qituvchi to'qnashuvi" in e for e in v["errors"]))

# ---------------------------------------------------------------- TEST 3
print("\nTEST 3: xona to'qnashuvi bo'lmasin")
d3 = {k: (list(x) if isinstance(x, list) else x) for k, x in d.items()}
d3["rooms"] = [{"id": "r1", "name": "Lab"}]
d3["assignments"] = [dict(a) for a in d["assignments"]]
for a in d3["assignments"]:
    if a["subjectId"] == "s0":
        a["roomId"] = "r1"
r3 = S.solve_safe(d3, max_seconds=12)
v3 = S.validate_schedule(r3["entries"], d3)
check("xona to'qnashuvi yo'q", not any("Xona to'qnashuvi" in e for e in v3["errors"]))

# ---------------------------------------------------------------- TEST 4
print("\nTEST 4: metodik kunda dars bo'lmasin")
d4 = {k: (list(x) if isinstance(x, list) else x) for k, x in d.items()}
d4["teachers"] = [dict(t) for t in d["teachers"]]
d4["teachers"][0]["methodicalDay"] = 2
r4 = S.solve_safe(d4, max_seconds=12)
bad = [e for e in r4["entries"] if e["teacherId"] == "t0" and e["day"] == 2]
check("metodik kunda dars yo'q", not bad, "%d ta buzilish" % len(bad))

# ---------------------------------------------------------------- TEST 5
print("\nTEST 5: ikki smena — REAL VAQT bo'yicha to'qnashuv")
d5 = {"school": school(shift_mode=2, lessons=6, lessonsPerDay2=6,
                       lessonDuration2=45, breakDuration2=10,
                       startShift2="14:00"),
      "rooms": [], "assignments": [],
      "teachers": [{"id": "t0", "name": "T", "maxHours": 40,
                    "methodicalDay": None, "unavailable": {}}],
      "subjects": [{"id": "s0", "name": "F", "difficulty": 5, "unavailable": {}}],
      "classes": [{"id": "c1", "name": "5-A", "shift": 1, "busyDays": {}},
                  {"id": "c2", "name": "5-B", "shift": 2, "busyDays": {}}]}
for ci in ("c1", "c2"):
    d5["assignments"].append({"id": "a" + ci, "teacherId": "t0", "subjectId": "s0",
                              "classId": ci, "hoursPerWeek": 5, "roomId": None,
                              "isSplit": False})
r5 = S.solve_safe(d5, max_seconds=12)
v5 = S.validate_schedule(r5["entries"], d5)
check("smenalararo soxta to'qnashuv yo'q", v5["valid"], str(v5["errors"][:1]))
check("ikkala smena ham joylashdi", len(r5["entries"]) == 10,
      "%d/10" % len(r5["entries"]))
check("slots_overlap: 1-sm 1-soat va 2-sm 1-soat kesishmaydi",
      not S.slots_overlap(d5["school"], 1, 0, 2, 0))

# ---------------------------------------------------------------- TEST 6
print("\nTEST 6: bo'linadigan A/B guruh bir slotda")
d6 = {"school": school(), "rooms": [], "assignments": [],
      "teachers": [{"id": "tA", "name": "A", "maxHours": 30, "methodicalDay": None, "unavailable": {}},
                   {"id": "tB", "name": "B", "maxHours": 30, "methodicalDay": None, "unavailable": {}}],
      "subjects": [{"id": "s0", "name": "Ingliz", "difficulty": 5, "unavailable": {}}],
      "classes": [{"id": "c1", "name": "5-A", "shift": 1, "busyDays": {}}]}
d6["assignments"].append({"id": "a1", "teacherId": "tA", "subjectId": "s0",
                          "classId": "c1", "hoursPerWeek": 3, "roomId": None,
                          "isSplit": True, "splitTeacherId": "tB", "splitRoomId": None})
r6 = S.solve_safe(d6, max_seconds=12)
slots = collections.defaultdict(set)
for e in r6["entries"]:
    slots[(e["day"], e["lesson"])].add(e.get("group"))
ok6 = all(g == {"A", "B"} for g in slots.values()) and len(slots) == 3
check("A va B har doim bir slotda", ok6, "%d slot" % len(slots))
check("validator: valid", S.validate_schedule(r6["entries"], d6)["valid"])

# ---------------------------------------------------------------- TEST 7
print("\nTEST 7: juft dars ketma-ket")
d7 = {"school": school(), "rooms": [], "assignments": [],
      "teachers": [{"id": "t0", "name": "T", "maxHours": 30, "methodicalDay": None, "unavailable": {}}],
      "subjects": [{"id": "s0", "name": "Texnologiya", "difficulty": 3,
                    "unavailable": {}, "doubleLesson": True}],
      "classes": [{"id": "c1", "name": "5-A", "shift": 1, "busyDays": {}}]}
d7["assignments"].append({"id": "a1", "teacherId": "t0", "subjectId": "s0",
                          "classId": "c1", "hoursPerWeek": 2, "roomId": None,
                          "isSplit": False})
r7 = S.solve_safe(d7, max_seconds=12)
byday = collections.defaultdict(list)
for e in r7["entries"]:
    byday[e["day"]].append(e["lesson"])
pairs = [sorted(v) for v in byday.values() if len(v) == 2]
check("juft dars ketma-ket", bool(pairs) and all(p[1] - p[0] == 1 for p in pairs),
      str(dict(byday)))

# ---------------------------------------------------------------- TEST 8
print("\nTEST 8: qulflangan darslar siljimasin")
d8 = {"school": school(), "rooms": [], "assignments": [],
      "teachers": [{"id": "t0", "name": "T", "maxHours": 30, "methodicalDay": None, "unavailable": {}}],
      "subjects": [{"id": "s0", "name": "F", "difficulty": 5, "unavailable": {}}],
      "classes": [{"id": "c1", "name": "5-A", "shift": 1, "busyDays": {}},
                  {"id": "c2", "name": "5-B", "shift": 1, "busyDays": {}}],
      "fixedEntries": [{"classId": "c2", "subjectId": "s0", "teacherId": "t0",
                        "roomId": None, "day": 0, "lesson": 0, "group": None}]}
d8["assignments"].append({"id": "a1", "teacherId": "t0", "subjectId": "s0",
                          "classId": "c1", "hoursPerWeek": 4, "roomId": None,
                          "isSplit": False})
r8 = S.solve_safe(d8, max_seconds=12)
kept = [e for e in r8["entries"]
        if e["classId"] == "c2" and e["day"] == 0 and e["lesson"] == 0]
moved = [e for e in r8["entries"]
         if e["classId"] == "c1" and e["day"] == 0 and e["lesson"] == 0]
check("qulflangan dars joyida qoldi", len(kept) == 1)
check("qulflangan dars ustiga qo'yilmadi", len(moved) == 0)

# ---------------------------------------------------------------- TEST 9
print("\nTEST 9: imkonsiz jadval — qulamasin, sabab qaytsin")
d9 = {"school": school(days=5, lessons=2), "rooms": [], "assignments": [],
      "teachers": [{"id": "t0", "name": "Aliyev", "maxHours": 4,
                    "methodicalDay": None, "unavailable": {}}],
      "subjects": [{"id": "s0", "name": "F", "difficulty": 5, "unavailable": {}}],
      "classes": [{"id": "c1", "name": "5-A", "shift": 1, "busyDays": {}}]}
d9["assignments"].append({"id": "a1", "teacherId": "t0", "subjectId": "s0",
                          "classId": "c1", "hoursPerWeek": 10, "roomId": None,
                          "isSplit": False})
try:
    r9 = S.solve_safe(d9, max_seconds=10)
    crashed = False
except Exception as exc:                                    # noqa: BLE001
    crashed = True
    r9 = None
    print("     QULADI:", exc)
check("qulamadi", not crashed)
if r9:
    u = r9.get("unfilled") or []
    check("unfilled qaytdi", bool(u), "%d yozuv" % len(u))
    check("sabab ko'rsatildi", bool(u and u[0].get("reason")),
          (u[0].get("reason", "")[:60] if u else ""))
    check("maxHours AVTOMATIK buzilmadi",
          all(sum(1 for e in r9["entries"] if e["teacherId"] == "t0") <= 4
              for _ in [0]),
          "%d soat qo'yildi (limit 4)" % sum(1 for e in r9["entries"] if e["teacherId"] == "t0"))

# ---------------------------------------------------------------- TEST 10
print("\nTEST 10: rasmda ko'ringan holat — oxirgi dars oynadan keyin qolmasin")
d10 = {"school": school(days=6, lessons=6), "rooms": [], "assignments": [],
       "teachers": [{"id": "t0", "name": "Sinf rahbari", "maxHours": 30,
                     "methodicalDay": None, "unavailable": {}},
                    {"id": "t1", "name": "Rus tili", "maxHours": 30,
                     "methodicalDay": None, "unavailable": {}}],
       "subjects": [{"id": "s0", "name": "Matematika", "difficulty": 9, "unavailable": {}},
                    {"id": "s1", "name": "Ona tili", "difficulty": 8, "unavailable": {}},
                    {"id": "s2", "name": "O'qish", "difficulty": 6, "unavailable": {}},
                    {"id": "s3", "name": "Tarbiya", "difficulty": 3, "unavailable": {}},
                    {"id": "s4", "name": "Rus tili", "difficulty": 5, "unavailable": {}}],
       "classes": [{"id": "c1", "name": "1-A", "shift": 1, "busyDays": {}}]}
for sid, h, tid in [("s0", 5, "t0"), ("s1", 5, "t0"), ("s2", 4, "t0"),
                    ("s3", 1, "t0"), ("s4", 2, "t1")]:
    d10["assignments"].append({"id": "a" + sid, "teacherId": tid, "subjectId": sid,
                               "classId": "c1", "hoursPerWeek": h, "roomId": None,
                               "isSplit": False})
r10 = S.solve_safe(d10, max_seconds=15)
g10 = gaps_of(r10["entries"], d10)
tot10 = sum(a["hoursPerWeek"] for a in d10["assignments"])
check("hamma dars joylashdi", len(r10["entries"]) == tot10,
      "%d/%d" % (len(r10["entries"]), tot10))
check("BO'SH OYNA YO'Q", g10 == 0, "%d oyna" % g10)
check("validator: valid", r10["validation"]["valid"], str(r10["validation"]["errors"][:1]))
per_day = collections.Counter(e["day"] for e in r10["entries"])
spread = max(per_day.values()) - min(per_day.values()) if len(per_day) == 6 else 99
check("kunlar teng taqsimlandi (farq <= 1)", spread <= 1,
      "farq %s, kunlar %s" % (spread, dict(sorted(per_day.items()))))

# ---------------------------------------------------------------- YAKUN
print("\n" + "=" * 62)
print("O'TDI: %d | O'TMADI: %d" % (len(PASS), len(FAIL)))
if FAIL:
    print("O'tmagan testlar:")
    for f in FAIL:
        print("  -", f)
print("=" * 62)
