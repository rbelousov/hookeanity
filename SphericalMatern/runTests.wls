(* Run paclet tests *)
repoDir = DirectoryName @ $InputFileName;
PacletDirectoryLoad @ DirectoryName[repoDir];
testDir = FileNameJoin[{repoDir, "Tests"}];
report = TestReport[FileNames["*.wl", testDir]];

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
