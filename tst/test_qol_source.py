"""Source-level guards for the selective QOL port; no build or binary is invoked.

These checks cover call placement and retained guards, not numerical correctness,
MPI completion, or pgen reentrancy. Runtime coverage must separately check repeated
startup refinement, direct fine-grid ICs, restart preservation, boundary/C2P state,
all 32 history values, and CLI insertion/overwrite/missing-block rejection.
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def source(path):
    """Read source with comments removed so comments cannot satisfy assertions."""
    text = (ROOT / path).read_text(encoding="utf-8")
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.DOTALL)


def body(text, marker):
    """Extract a braced block after a unique marker in the inspected C++ regions."""
    start = text.index("{", text.index(marker))
    depth = 1
    for end in range(start + 1, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start + 1:end]
    raise AssertionError(f"Unclosed block after {marker}")


class QOLSourceTests(unittest.TestCase):
    def test_startup_is_fresh_only_and_precedes_outputs(self):
        main = source("src/main.cpp")
        startup = body(main, "if (pmesh->adaptive && !res_flag)")
        loop = body(startup, "while (true)")
        refresh = "pdriver->InitBoundaryValuesAndPrimitives(pmesh);"
        self.assertLess(startup.index(refresh), startup.index("while (true)"))
        calls = [
            "CheckForRefinement(pmesh->pmb_pack, true)",
            "UpdateMeshBlockTree(nnew, ndel)",
            "if (nnew == 0 && ndel == 0) {break;}",
            "RedistAndRefineMeshBlocks(pinput, nnew, ndel)",
            "MarkMeshUpdated()",
            "CallProblemGenerator(pinput, false)",
            refresh,
        ]
        positions = [loop.index(call) for call in calls]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(main.count("CallProblemGenerator("), 1)
        self.assertEqual(loop.count("MarkMeshUpdated()"), 1)
        self.assertNotIn("make_unique<ProblemGenerator>", startup)
        self.assertIn("nmb_created += nnew", loop)
        self.assertIn("nmb_deleted += ndel", loop)
        self.assertLess(main.index("if (pmesh->adaptive && !res_flag)"),
                        main.index("Outputs* pout = new Outputs"))
        self.assertNotIn("AssembleTaskLists", main)
        self.assertLess(main.index("AddCoordinatesAndPhysics(pinput)"),
                        main.index("std::make_unique<ProblemGenerator>"))

    def test_only_initial_pass_bypasses_runtime_cadence(self):
        header = source("src/mesh/mesh_refinement.hpp")
        self.assertIn("bool initial_pass = false", header)
        refinement = source("src/mesh/mesh_refinement.cpp")
        check = body(refinement, "void MeshRefinement::CheckForRefinement(")
        first = check.index("if (!initial_pass)")
        second = check.index("if (!initial_pass)", first + 1)
        cadence = body(check, "if (!initial_pass)")
        cooldown = body(check[second:], "if (!initial_pass)")
        self.assertIn("ncyc_since_ref(m) += 1", cadence)
        self.assertIn("(pmy_mesh->ncycle)%(ncyc_check_amr)", cadence)
        self.assertIn("ncyc_since_ref(m+mbs) < refinement_interval", cooldown)
        self.assertIn("pmy_mesh->max_level", check)
        self.assertIn("pmy_mesh->root_level", check)
        self.assertIn("MPI_Allgatherv", check)
        redist = body(refinement, "void MeshRefinement::RedistAndRefineMeshBlocks(")
        self.assertIn("Kokkos::realloc(ncyc_since_ref, new_nmb_total)", redist)
        self.assertIn("Kokkos::deep_copy(ncyc_since_ref, new_ncyc_since_ref)", redist)

    def test_runtime_amr_retains_notification_repair_and_refresh(self):
        runtime = body(source("src/mesh/mesh_refinement.cpp"),
                       "void MeshRefinement::AdaptiveMeshRefinement(")
        refresh = "pdriver->InitBoundaryValuesAndPrimitives(pmy_mesh);"
        self.assertIn("CheckForRefinement(pmy_mesh->pmb_pack);", runtime)
        self.assertLess(runtime.index("MarkMeshUpdated()"), runtime.index(refresh))
        self.assertLess(runtime.index(refresh), runtime.index("RepairAMRFC("))
        self.assertLess(runtime.index("RepairAMRFC("), runtime.rindex(refresh))
        self.assertNotIn("CallProblemGenerator", runtime)
        tree = source("src/mesh/build_tree.cpp")
        rank_guard = body(tree, "if (nmb_total < global_variable::nranks)")
        self.assertIn("std::exit(EXIT_FAILURE)", rank_guard)

    def test_history_filename_labels_and_paired_capacity(self):
        history = source("src/outputs/history.cpp")
        spacetime = history.split("case PhysicsModule::SpaceTimeDynamics:", 1)[1]
        spacetime = spacetime.split("case PhysicsModule::UserDefined:", 1)[0]
        self.assertIn('fname.append(".z4c");', spacetime)
        self.assertIn("break;", spacetime)
        self.assertIn('"[%d]=%-16s "', history)
        self.assertNotIn("%.10s", history)
        for path, macro in [("src/outputs/outputs.hpp", "NHISTORY_VARIABLES"),
                            ("src/athena.hpp", "NREDUCTION_VARIABLES")]:
            self.assertRegex(source(path), rf"#define\s+{macro}\s+32\b")
        self.assertIn("#if NHISTORY_VARIABLES > NREDUCTION_VARIABLES",
                      source("src/outputs/outputs.hpp"))

    def test_cli_inserts_only_inside_existing_blocks(self):
        cli = body(source("src/parameter_input.cpp"),
                   "void ParameterInput::ModifyFromCmdline(")
        missing_block = body(cli, "if (pb == nullptr)")
        missing_parameter = body(cli, "if (pl == nullptr)")
        self.assertIn("std::exit(EXIT_FAILURE)", missing_block)
        self.assertIn('AddParameter(pb, name, value, "");', missing_parameter)
        self.assertNotIn("std::exit", missing_parameter)
        self.assertLess(cli.index("if (pb == nullptr)"), cli.index("AddParameter("))
        self.assertIn("pl->param_value.assign(value)", cli)
        self.assertIn("pb->max_len_parvalue = value.length()", cli)
        self.assertNotIn("FindOrAddBlock", cli)
        self.assertIn("bool from_code = false", source("src/parameter_input.hpp"))


if __name__ == "__main__":
    unittest.main()
