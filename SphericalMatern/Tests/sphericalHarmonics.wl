(* Tests for SphericalMatern`SphericalHarmonics and SphericalMatern`FibonacciPoints *)

Needs["SphericalMatern`"];

VerificationTest[
  Module[{n = 100, nodes, weight, basis, vals, tol = 0.1, ok},

    nodes = FibonacciPoints[n];
    weight = N[4 * Pi / Length[nodes]];
    basis = Flatten[Table[{ell, m}, {ell, 0, 6}, {m, -ell, ell}], 1];

    (* Discrete inner products *)
    vals = Table[
      With[{
        ell1 = basis[[i, 1]], m1 = basis[[i, 2]],
        ell2 = basis[[j, 1]], m2 = basis[[j, 2]]
        },
        weight * Total[
          Map[
            RealSphericalHarmonic[ell1, m1, Sequence @@ #] *
            RealSphericalHarmonic[ell2, m2, Sequence @@ #] &,
            nodes
          ]
        ]
      ],
      {i, Length[basis]}, {j, Length[basis]}
    ];

    (* Check approximate orthonormality with absolute tolerance tol *)
    ok = And @@ Flatten[
        Table[
            Abs[vals[[i, j]] - If[i == j, 1, 0]] <= tol,
            {i, Length[basis]}, {j, Length[basis]}
        ], 1
    ];

    ok
  ],
  True,
  TestID -> "Orthonormal spherical harmonics over Fibonacci quadrature (l<=6, tol=0.1)"
]
