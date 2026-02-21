(* Install paclet *)
Needs["PacletTools`"];

pacletName = "SphericalMatern";
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

(* --- Uninstall previous versions if present --- *)
existingPaclets = PacletFind[pacletName];
If[existingPaclets =!= {},
  Print["Existing installation(s) found. Uninstalling..."];
  Scan[PacletUninstall, existingPaclets];
];

(* --- Install new version --- *)
installResult = PacletInstall[pacletArchive];
Print["Built: ", pacletArchive];
Print["Installed: ", installResult];
Print["PacletFind: ", PacletFind[pacletName]];