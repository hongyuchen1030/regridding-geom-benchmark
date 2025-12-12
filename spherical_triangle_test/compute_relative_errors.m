#!/global/common/software/nersc9/mathematica/13.0.1/bin/math -script
(* compute_relative_errors.m
   ARGS: <geom_csv> <methods_csv>
   Output: next to methods_csv as error_minangle_all<input_file_name>.csv
*)

(* ---------- precision knobs ---------- *)
PREC     = 17;
WORKPREC = 80;

(* ---------- helpers ---------- *)
toInt[x_] := Which[ IntegerQ[x], x, StringQ[x], ToExpression[x], True, Round@N@x ];

restoreBinSigExp[s_, e_] := Module[{si = toInt[s], ex = toInt[e]},
  N[ SetPrecision[si, Infinity] * 2^ex, WORKPREC]
];

normalize[v_] := v / Sqrt[v.v];

triAnglesArea[v0_, v1_, v2_] := Module[
  {p0, p1, p2, u01, u12, u20, ca1, ca2, ca3, a1, a2, a3},
  p0 = SetPrecision[v0, WORKPREC];
  p1 = SetPrecision[v1, WORKPREC];
  p2 = SetPrecision[v2, WORKPREC];
  u01 = normalize[Cross[p0, p1]];
  u12 = normalize[Cross[p1, p2]];
  u20 = normalize[Cross[p2, p0]];
  ca1 = -u01.u20; ca2 = -u12.u01; ca3 = -u20.u12;
  a1 = ArcCos[ca1]; a2 = ArcCos[ca2]; a3 = ArcCos[ca3];
  {{a1, a2, a3}, a1 + a2 + a3 - Pi}
];

toDecSigExp[x_] := Module[{m, e}, {m, e} = MantissaExponent[N[Abs[x], WORKPREC], 10]; {N[m, PREC], e}];

say[s_] := WriteString[$Output, s <> "\n"];

(* ---------- robust arg parsing ---------- *)
debugQ = (Environment["DEBUG_MATH"] === "1");

safeRest[list_] := If[ListQ[list] && Length[list] > 0, Rest[list], {}];

getArgs[] := Module[{sc, cl, i, after, geom, meth, csvs},
  sc = safeRest[$ScriptCommandLine];
  If[Length[sc] >= 2 && And @@ (StringQ /@ sc[[;; 2]]), Return[sc[[;; 2]]]];

  cl = $CommandLine;
  i = FirstPosition[cl, "-script", Missing["nf"]];
  If[i =!= Missing["nf"],
    after = Drop[cl, i[[1]]];
    If[Length[after] >= 3,
      geom = after[[2]]; meth = after[[3]];
      If[StringQ[geom] && StringQ[meth], Return[{geom, meth}]];
    ];
  ];

  csvs = Select[cl, StringQ[#] && StringMatchQ[#, ___ ~~ ".csv"] &];
  If[Length[csvs] >= 2, Return[csvs[[;; 2]]]];

  {}
];

If[debugQ,
  say["[Debug] $ScriptCommandLine=" <> ToString[$ScriptCommandLine]];
  say["[Debug] $CommandLine=" <> ToString[$CommandLine]];
];

args = getArgs[];

If[debugQ, say["[Debug] args=" <> ToString[args]]];

If[Length[args] < 2,
  say["Usage: compute_relative_errors.m <geom_csv> <methods_csv>"];
  say["  <geom_csv>:     full path to triangles CSV (sig/exp base-2)"];
  say["  <methods_csv>:  full path to methods CSV (methods_area_<input_file_name>.csv)"];
  Exit[1];
];

geomCSV = args[[1]];
methCSV = args[[2]];

If[!FileExistsQ[geomCSV], say["[FATAL] geometry CSV not found: " <> ToString[geomCSV]]; Exit[2]];
If[!FileExistsQ[methCSV], say["[FATAL] methods CSV not found: " <> ToString[methCSV]]; Exit[3]];

methodsBase = FileBaseName[methCSV];
inputTag = StringReplace[methodsBase, {"methods_area_" -> "", ".csv" -> ""}];
outDir = DirectoryName[methCSV];
outCSV = FileNameJoin[{outDir, "error_minangle_all" <> inputTag <> ".csv"}];

(* ---------- load geometry ---------- *)
geomAssoc = Association[];
rawG = Import[geomCSV, "CSV"];
If[Length[rawG] < 2, say["[ERROR] empty geom CSV: " <> geomCSV]; Exit[4]];
rowsG = Rest[rawG];
malformedG = Select[rowsG, Length[#] =!= 19 &];
If[malformedG =!= {}, say["[WARN] malformed geometry rows encountered; skipping malformed."]];

rowsG // Scan[
  Function[r,
    If[Length[r] == 19,
      Module[{fid, v0, v1, v2},
        fid = toInt[r[[1]]];
        v0 = { restoreBinSigExp[r[[2]],r[[3]]],  restoreBinSigExp[r[[4]],r[[5]]],  restoreBinSigExp[r[[6]],r[[7]]]  };
        v1 = { restoreBinSigExp[r[[8]],r[[9]]],  restoreBinSigExp[r[[10]],r[[11]]], restoreBinSigExp[r[[12]],r[[13]]] };
        v2 = { restoreBinSigExp[r[[14]],r[[15]]], restoreBinSigExp[r[[16]],r[[17]]], restoreBinSigExp[r[[18]],r[[19]]] };
        v0 = v0 / Sqrt[v0.v0]; v1 = v1 / Sqrt[v1.v1]; v2 = v2 / Sqrt[v2.v2];
        geomAssoc[fid] = {v0, v1, v2};
      ]
    ]
  ]
];

If[Length[geomAssoc] == 0, say["[FATAL] no valid triangles parsed from " <> geomCSV]; Exit[5]];

(* ---------- baseline (now includes h) ---------- *)
baselineAssoc = Association[];
Keys[geomAssoc] // Scan[
  Function[id,
    Module[{v0, v1, v2, angs, area, minang, h},
      {v0, v1, v2} = geomAssoc[id];
      {angs, area} = triAnglesArea[v0, v1, v2];
      minang = Min[angs];
      h = Max[Norm[v0 - v1], Norm[v1 - v2], Norm[v2 - v0]];
      baselineAssoc[id] = {N[minang, PREC], N[area, WORKPREC], N[h, PREC]};
    ]
  ]
];
say["[Info] baseline faces: " <> ToString[Length[baselineAssoc]]];

(* ---------- methods (dynamic header parsing) ---------- *)

(* Build method name/order and column maps from header *)
getMethodHeaderMaps[hdr_List] := Module[
  {withPos, sigRules, expRules, sigAssoc, expAssoc, orderedMethods},

  withPos = MapIndexed[{#1, #2[[1]]} &, hdr];

  sigRules = Map[
    With[{h = #[[1]], i = #[[2]]},
      If[StringEndsQ[h, "_sig"], StringReplace[h, "_sig" -> ""] -> i, Nothing]
    ] &,
    withPos
  ];
  expRules = Map[
    With[{h = #[[1]], i = #[[2]]},
      If[StringEndsQ[h, "_exp"], StringReplace[h, "_exp" -> ""] -> i, Nothing]
    ] &,
    withPos
  ];

  sigAssoc = Association@sigRules;
  expAssoc = Association@expRules;

  (* Order methods by the position of their _sig column as they appear in the CSV *)
  orderedMethods = Keys @ KeySortBy[sigAssoc, sigAssoc];

  {orderedMethods, sigAssoc, expAssoc}
];

(* Global method list (filled by parseMethods) *)
methodNames = {};

parseMethods[mfile_] := Module[{raw, hdr, rows, assoc = Association[], maps, sigMap, expMap},
  raw = Import[mfile, "CSV"];
  If[Length[raw] < 2, Return[assoc]];
  hdr  = First[raw];
  rows = Rest[raw];

  (* Discover methods and column indices *)
  {methodNames, sigMap, expMap} = getMethodHeaderMaps[hdr];

  rows // Scan[
    Function[r,
      Module[{fid, m},
        (* Safety: face_id assumed to be the first column *)
        If[Length[r] >= 1,
          fid = toInt[r[[1]]];
          m = Association@
                Table[
                  With[{nm = nm, sIdx = sigMap[nm], eIdx = expMap[nm]},
                    If[IntegerQ[sIdx] && IntegerQ[eIdx] && sIdx <= Length[r] && eIdx <= Length[r],
                      nm -> restoreBinSigExp[r[[sIdx]], r[[eIdx]]],
                      nm -> Missing["col"]
                    ]
                  ],
                  {nm, methodNames}
                ];
          assoc[fid] = m;
        ];
      ]
    ]
  ];
  assoc
];

methodsAssoc = parseMethods[methCSV];
say["[Info] methods faces: " <> ToString[Length[methodsAssoc]]];
say["[Info] methods detected: " <> StringRiffle[ToString /@ methodNames, ", "]];


(* ---------- rows ---------- *)
makeRow[id_] := Module[{minang, area, h, mvals, relErrPairs},
  If[!KeyExistsQ[baselineAssoc, id] || !KeyExistsQ[methodsAssoc, id], Return[Nothing]];
  {minang, area, h} = baselineAssoc[id];
  mvals = methodsAssoc[id];
  relErrPairs = Flatten @ (toDecSigExp[(mvals[#] - area)/area] & /@ methodNames);
  Join[{id, N[minang, PREC]}, relErrPairs, {N[h, PREC]}]
];

allIds = Intersection[Keys[baselineAssoc], Keys[methodsAssoc]];
rows    = DeleteCases[makeRow /@ allIds, Nothing];
rowsSorted = SortBy[rows, #[[2]] &];

header = Join[
  {"face_id", "min_angle"},
  Flatten[Table[{m <> "_err_mant10", m <> "_err_exp10"}, {m, ToString /@ methodNames}]],
  {"h"}
];

Export[outCSV, Prepend[rowsSorted, header], "CSV"];
say["[Wrote] " <> outCSV];
