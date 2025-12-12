#!/usr/bin/env -S math -script

resultsDir = "results";
arcsFile   = resultsDir <> "/generated_arcs.csv";
anglesFile = resultsDir <> "/angle_results.csv";
outFile    = resultsDir <> "/relative_errors.csv";

(* format helpers: 17 digits; pick one *)
fmt17[x_]    := ToString @ NumberForm[x, {Infinity, 17}, NumberPadding -> {"", "0"}, NumberPoint -> "."];
fmt17Sci[x_] := ToString @ ScientificForm[x, 17, NumberFormat -> (Row[{#1, "e", #3}] &)];  (* "1.23e-16" style *)

(* ===== 1. Read generated_arcs.csv for baseline ===== *)
arcsRaw = Import[arcsFile, "CSV"]; arcsRows = Rest[arcsRaw];  

recon[sig_, exp_] := N[SetPrecision[sig, 17] * 2^exp, 17];

baselineTable =
  Map[
    Function[row,
      Module[{u, v, dot, crossNorm, baseRad, baseDeg},
        u = {recon[row[[2]], row[[3]]], recon[row[[4]], row[[5]]], recon[row[[6]], row[[7]]]};
        v = {recon[row[[8]], row[[9]]], recon[row[[10]], row[[11]]], recon[row[[12]], row[[13]]]};
        dot = N[u.v, 17];
        crossNorm = N[Norm[Cross[u, v]], 17];

        (* ArcTan[x,y] = atan2(y,x); we want atan2(cross, dot) -> ArcTan[dot, cross] *)
        baseRad = N[ArcTan[dot, crossNorm], 17];

        (* Use /Degree instead of 180./Pi to avoid dropping to machine precision *)
        baseDeg = N[baseRad / Degree, 17];

        {baseDeg, baseRad, u, v, dot, crossNorm}
      ]
    ],
    arcsRows
  ];


(* ===== 2. Read angle_results.csv ===== *)
anglesRaw = Import[anglesFile, "CSV"]; anglesRows = Rest[anglesRaw];

angleTable =
  Map[
    Function[row,
      { N[row[[1]], 17],
        recon[row[[2]], row[[3]]],
        recon[row[[4]], row[[5]]],
        recon[row[[6]], row[[7]]],
        recon[row[[8]], row[[9]]]
      }
    ],
    anglesRows
  ];

(* ===== 3. Sort (still here to keep your flow) ===== *)
baselineSorted = SortBy[baselineTable, First];
anglesSorted   = SortBy[angleTable,   First];

If[Length[baselineSorted] != Length[anglesSorted],
  Print["ERROR: Row count mismatch!"]; Quit[1];
];



(* ===== 4. Compute relative errors (unchanged numerically) ===== *)
resultsNumeric =
  MapThread[
    Function[{baseRow, angleRow},
      Module[{degBase, base, recArctan, recKahan, recAsin, recAcos},
        degBase    = baseRow[[1]];
        base       = baseRow[[2]];
        recArctan  = angleRow[[2]];
        recKahan   = angleRow[[3]];
        recAsin    = angleRow[[4]];
        recAcos    = angleRow[[5]];
        {degBase,
         (recArctan - base)/base,
         (recKahan  - base)/base,
         (recAsin   - base)/base,
         (recAcos   - base)/base}
      ]
    ],
    {baselineSorted, anglesSorted}
  ];

(* ===== 5. Export with the SAME formatting you see in prints ===== *)
resultsForCSV =
  Prepend[
    Map[N[#, 17] &, resultsNumeric, {2}],
    {"baseline_deg", "err_arctan", "err_kahan", "err_asin", "err_acos"}
  ];

Export[outFile, resultsForCSV, "CSV"];
Print["\nRelative errors written to: ", outFile];
