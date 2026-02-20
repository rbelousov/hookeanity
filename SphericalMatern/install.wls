(* install.wls *)
Needs["PacletTools`"];

pacletDir = DirectoryName[$InputFileName];

buildResult = PacletBuild[pacletDir];

If[Head[buildResult] =!= Success,
  Print["PacletBuild failed: ", buildResult];
  Exit[1];
];

pacletArchive = Lookup[buildResult[[2]], "PacletArchive", Missing["NoPacletArchive"]];

If[pacletArchive === Missing["NoPacletArchive"] || !FileExistsQ[pacletArchive],
  Print["PacletBuild did not produce a paclet archive path. Result: ", buildResult];
  Exit[1];
];

installResult = PacletInstall[pacletArchive];

Print["Built: ", pacletArchive];
Print["Installed: ", installResult];

(* Optional sanity check *)
Print["PacletFind: ", PacletFind["SphericalMatern"]];
