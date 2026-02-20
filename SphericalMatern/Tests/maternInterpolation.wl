(* Tests for SphericalMatern`MaternInterpolation *)

Needs["SphericalMatern`"];

VerificationTest[
  Module[{fnc, angles, values, inter, points, ok, tol = 0.1},

    fnc[{theta_, phi_}] := BesselJ[0, Cos[theta] * Sin[phi]];
    angles = FibonacciPoints[100];
    values = fnc /@ angles;

    inter = MaternInterpolation[angles, values];
    SeedRandom@0;
    points = RandomSample[angles, 10];
    
    ok = And @@ (
        Function[pnt, Abs[inter @@ pnt - fnc[pnt]] <= tol] /@ points
    );

    ok
  ],
  True,
  TestID -> "Matern interpolation (tol=0.1)"
]

VerificationTest[
  Module[{fnc, angles, values, inter, points, ok, tol = 0.1},

    fnc[{theta_, phi_}] := BesselJ[0, Cos[theta] * Sin[phi]];
    angles = FibonacciPoints[100];
    values = fnc /@ angles;

    inter = MaternInterpolation[angles, values];
    SeedRandom@0;
    points = Table[{RandomReal[{0, Pi}], RandomReal[{0, 2 * Pi}]}, 10];
    
    ok = And @@ (
        Function[pnt, Abs[inter @@ pnt - fnc[pnt]] <= tol] /@ points
    );

    ok
  ],
  True,
  TestID -> "Matern interpolation (tol=0.1)"
]
