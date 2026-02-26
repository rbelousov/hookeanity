(* ::Package:: *)

BeginPackage["SphericalMatern`"];

RealSphericalHarmonic::usage =
"RealSphericalHarmonic[l, m, \[Theta], \[Phi]] evaluates to an orthonormalized real-valued spherical harmonic \
of degree l and order m at the polar angle \[Theta] and azimuthal angle \[Phi]. \
Here l is a nonnegative integer, m is an integer with -l \[LessEqual] m \[LessEqual] l.";

RealSphericalHarmonic::arg =
"Expected l to be a nonnegative integer and m to be an integer with -l <= m <= l.";

FibonacciPoints::usage =
"FibonacciPoints[n] spreads n > 0 points (\[Theta], \[Phi]) nearly uniformly over the spherical surface \
avoiding the northern and southern poles."

FibonacciPoints::arg = "Expected n to be a positive integer.";

MaternInterpolation::usage =
"MaternInterpolation[angles, values] constructs a Matérn-type spherical kernel interpolant \
from samples at spherical angles.\n\n\
* angles is a list of {\[Theta], \[Phi]} pairs (in radians).\n\
* values is a list of the same length containing sample values.\n\
The result is a function f[\[Theta], \[Phi]] that evaluates the interpolant.\n\n\
MaternInterpolation[angles, values, smoothness] uses the given smoothness parameter (default 3).";

MaternKernel::usage =
"MaternKernel[{\[Theta], \[Phi]}, order, smoothness] constructs a Matérn-type spherical kernel function \
centered at the given solid angle {\[Theta], \[Phi]} with a specified order > 0 and smoothness (default 3).";
MaternKernel::arg = "Expected numeric {\[Theta], \[Phi]} and positive integer order.";
MaternKernel::smooth = "Expected smoothness to be a positive number.";

MaternInterpolation::arg =
"Expected angles to be a nonempty list of {\[Theta], \[Phi]} pairs and values to be a list of the same length.";
MaternInterpolation::smooth = MaternKernel::smooth;

Begin["`Private`"];

(* Orthnormal real spherical harmonics *)
RealSphericalHarmonic[
  l_Integer?NonNegative, m_Integer,
  \[Theta]_, \[Phi]_
] /; -l <= m <= l :=
 Module[{am = Abs[m]},
  (-1)^m * Sqrt[
      (2 * l + 1) / (4 * Pi) * Factorial[l - am]/Factorial[l + am]
    ] *
   LegendreP[l, am, Cos[\[Theta]]] *
   Which[
     m > 0, Sqrt[2] * Cos[m * \[Phi]],
     m < 0, Sqrt[2] * Sin[-m * \[Phi]],
     True, 1
   ]
 ];

(* Friendly message for invalid integer constraints *)
RealSphericalHarmonic[l_, m_, \[Theta]_, \[Phi]_] /;
   !(IntegerQ[l] && l >= 0 && IntegerQ[m] && -l <= m <= l) :=
  (Message[RealSphericalHarmonic::arg]; $Failed);


(* Fibonacci points *)
FibonacciPoints[n_Integer?Positive] := Module[
  {k, gr, goldenAngle, z, theta, phi},
  
  (* k = 0, 1, ..., n-1 *)
  k = Range[0, n - 1];
  
  (* Golden ratio and golden angle *)
  gr = (1 + Sqrt[5]) / 2;
  goldenAngle = 2 * Pi / (gr^2);
  
  (* Midpoint z-spacing to avoid poles *)
  z = 1 - (2 * k + 1) / n;
  
  theta = N@ArcCos[z];
  phi = N@Mod[(k + 1 / 2) goldenAngle, 2 * Pi];
  
  Transpose[{theta, phi}]
];

FibonacciPoints[n_] /; !(IntegerQ[n] && n > 0) :=
  (Message[FibonacciPoints::arg]; $Failed);

ValidSolidAnglesQ[angles_] := MatrixQ[angles, NumericQ] && Dimensions[angles][[2]] === 2;
ValidScalarValuesQ[values_] := VectorQ[values, NumericQ];

MaternKernel[{\[Theta]_, \[Phi]_}, order_, smoothness_: 3] := Module[
  {grain, lmax, bandfilter},

  If[!(NumericQ[\[Theta]] && NumericQ[\[Phi]] && IntegerQ[order] && order > 0),
    Message[MaternKernel::arg];
    Return[$Failed];
  ];

  If[!(NumericQ[smoothness] && smoothness >= 0),
    Message[MaternKernel::smooth];
    Return[$Failed];
  ];

  grain = 4 * Pi / order;
  lmax = Floor @ Sqrt[order];
  bandfilter = Table[
    (1 + l * (l + 1) * grain) ^ -(1 + smoothness),
    {l, 0, lmax}
  ];

  Function[{theta, phi},
    Sum[
      Times[
        bandfilter[[l + 1]] * (2 * l + 1) / (4 * Pi),
        LegendreP[l, Cos[\[Theta]] * Cos[theta] + Sin[\[Theta]] * Sin[theta] * Cos[\[Phi] - phi]]
      ],
      {l, 0, lmax}
    ]
  ]
]

MaternInterpolation[angles_, values_, smoothness_: 3] :=  Module[
  {sz, grain, lmax, cth, sth, phis, cph, geo, bandfilter, mtx, expansion},

  If[!(ValidSolidAnglesQ[angles] && ValidScalarValuesQ[values] && Length[angles] === Length[values]),
    Message[MaternInterpolation::arg];
    Return[$Failed];
  ];

  If[!(NumericQ[smoothness] && smoothness >= 0),
    Message[MaternInterpolation::smooth];
    Return[$Failed];
  ];
 
  sz = Length @ angles;
  grain = 4 * Pi / sz;
  lmax = Floor @ Sqrt[sz];

  cth = Cos @ angles[[All, 1]];
  sth = Sin @ angles[[All, 1]];
  phis = angles[[All, 2]];
  cph = Cos @ Outer[Subtract, phis, phis];
  geo = Outer[Times, cth, cth] + Outer[Times, sth, sth] * cph;

  bandfilter = Table[
    (1 + l * (l + 1) * grain) ^ -(1 + smoothness),
    {l, 0, lmax}
  ];

  mtx = ConstantArray[0., {sz, sz}];
  Do[
    mtx += (bandfilter[[l + 1]] * (2 * l + 1) /  (4 * Pi)) * LegendreP[l, geo],
    {l, 0, lmax}
  ];

  expansion = LinearSolve[mtx, values];

  Function[{theta, phi},
    expansion . Sum[
      Times[
        bandfilter[[l + 1]] * (2 l + 1) / (4 * Pi),
        LegendreP[l, cth * Cos[theta] + sth * Sin[theta] * Cos[phis - phi]]
      ],
      {l, 0, lmax}
    ]
  ]
];

End[];
EndPackage[];
