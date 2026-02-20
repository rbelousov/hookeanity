repoDir = Directory[];
testDir = FileNameJoin[{repoDir, "Tests"}];

PacletDirectoryLoad[repoDir]
report = TestReport[FileNames["*.wl", testDir]];

(* Extract summary information *)
passed   = report["TestsSucceededCount"];
failed   = report["TestsFailedCount"];
skipped  = report["TestsNotEvaluatedCount"];
errors   = report["TestsFailedWithMessagesCount"];
time     = N @ QuantityMagnitude @ report["AbsoluteTimeUsed"];

Print["===================================="];
Print["SphericalMatern Test Summary"];
Print["------------------------------------"];
Print["Total tests:     ", total];
Print["Passed:          ", passed];
Print["Failed:          ", failed];
Print["Message failures:", errors];
Print["Skipped:         ", skipped];
Print["------------------------------------"];
Print["Total time (s):  ", time];
Print["===================================="];

If[failed > 0 || errors > 0,
  Exit[1],
  Exit[0]
];
