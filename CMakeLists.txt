cmake_minimum_required(VERSION 3.16)
project(TDMASH LANGUAGES NONE)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME pt_cn COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p test_pt_cn.py -v)
    add_test(NAME propagation_regression COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p test_propagation_regression.py -v)
    add_test(NAME moving_basis_regression COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p test_moving_basis.py -v)
    add_test(NAME timestep_comparison_self_check COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_pt_cn_timestep.py" --self-check)
    add_test(NAME bomd_workflow_self_check COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/euler_pt_cn_bomd.py" self-check)
    add_test(NAME paired_timing_self_check COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/benchmark_pt_cn_bomd.py" self-check)
    add_test(NAME unrestricted_pt_cn COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p test_pt_unrestricted.py -v)
    add_test(NAME unrestricted_moving_basis COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p test_unrestricted_moving_basis.py -v)
    add_test(NAME c2h4_workflow_self_check COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/euler_c2h4_ehrenfest.py" self-check)
    add_test(NAME cn_reference_workflow_self_check COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/euler_c2h4_cn_reference.py" self-check)
    add_test(NAME mom_predictor_workflow_self_check COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/euler_c2h4_mom_recovery.py" --self-check)
    add_test(NAME mom_continuation_workflow_self_check COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/euler_c2h4_mom_continue.py" --self-check)
    add_test(NAME cn_pt_paired_timing_self_check COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/benchmark_c2h4_cn_pt.py" --self-check)
    add_test(NAME cn_long_extension_self_check COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/extend_c2h4_cn_reference.py" --self-check)
    add_test(NAME cayley_fixed_nuclei_self_check COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/euler_c2h4_cayley.py" self-check)
    add_test(NAME cayley_spectral_self_check COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/analyze_c2h4_cayley.py" --self-check)
    add_test(NAME cayley_collection_self_check COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/collect_c2h4_cayley.py" --self-check)
    add_test(NAME theory_four_regression COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p test_theory4.py -v)
    set_tests_properties(pt_cn propagation_regression moving_basis_regression timestep_comparison_self_check bomd_workflow_self_check paired_timing_self_check unrestricted_pt_cn unrestricted_moving_basis c2h4_workflow_self_check cn_reference_workflow_self_check mom_predictor_workflow_self_check mom_continuation_workflow_self_check cn_pt_paired_timing_self_check cn_long_extension_self_check cayley_fixed_nuclei_self_check cayley_spectral_self_check cayley_collection_self_check theory_four_regression PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;MPLBACKEND=Agg"
        LABELS "regression"
        TIMEOUT 120)
endif()
