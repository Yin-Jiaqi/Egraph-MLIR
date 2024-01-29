#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <regex>
#include <functional>
#include <cerrno>
#include <cassert>

#include "include/utilities.hpp"
#include "include/graph.hpp"
#include "include/parser.hpp"
#include "include/reverter.hpp"

const std::string inputText = R"(
    (func_func_none "%arg0" "%arg1" "%arg2" "%arg3" (block_block_none (arith_constant_f64_0 2.000000e-01 "Symbol4%cst") (arith_indexcast_i32index_0 "%arg1" "Symbol4%0") (arith_indexcast_i32index_0 "%arg0" "Symbol4%1") (affine_forcontrol_none_0 (affine_forvalue_none_0 0 (arith_indexcast_i32index_0 "%arg0" "Symbol4%1") 1 "Symbol4%arg4") (block_block_none (affine_forcontrol_none_0 (affine_forvalue_none_0 1 (- "%0" 1) (* 2 1) "Symbol4%arg5") (block_block_none (affine_forcontrol_none_0 (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") (block_block_none (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%2") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (- "%arg6" 1) "Symbol4%3") (arith_addf_f64_0 (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%2") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (- "%arg6" 1) "Symbol4%3") "Symbol4%4") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (+ "%arg6" 1) "Symbol4%5") (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%2") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (- "%arg6" 1) "Symbol4%3") "Symbol4%4") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (+ "%arg6" 1) "Symbol4%5") "Symbol4%6") (affine_load_f64_2 "%arg2" (+ "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%7") (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%2") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (- "%arg6" 1) "Symbol4%3") "Symbol4%4") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (+ "%arg6" 1) "Symbol4%5") "Symbol4%6") (affine_load_f64_2 "%arg2" (+ "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%7") "Symbol4%8") (affine_load_f64_2 "%arg2" (- "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%9") (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%2") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (- "%arg6" 1) "Symbol4%3") "Symbol4%4") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (+ "%arg6" 1) "Symbol4%5") "Symbol4%6") (affine_load_f64_2 "%arg2" (+ "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%7") "Symbol4%8") (affine_load_f64_2 "%arg2" (- "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%9") "Symbol4%10") (arith_mulf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%2") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (- "%arg6" 1) "Symbol4%3") "Symbol4%4") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (+ "%arg6" 1) "Symbol4%5") "Symbol4%6") (affine_load_f64_2 "%arg2" (+ "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%7") "Symbol4%8") (affine_load_f64_2 "%arg2" (- "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%9") "Symbol4%10") (arith_constant_f64_0 2.000000e-01 "Symbol4%cst") "Symbol4%11") (affine_store_f64_2 (arith_mulf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%2") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (- "%arg6" 1) "Symbol4%3") "Symbol4%4") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (+ "%arg6" 1) "Symbol4%5") "Symbol4%6") (affine_load_f64_2 "%arg2" (+ "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%7") "Symbol4%8") (affine_load_f64_2 "%arg2" (- "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%9") "Symbol4%10") (arith_constant_f64_0 2.000000e-01 "Symbol4%cst") "Symbol4%11") "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%Pseudo3"))) (affine_apply_none (affine_forvalue_none_0 1 (- "%0" 1) (* 2 1) "Symbol4%arg5") "{d0}->{d0+1}") (affine_forcontrol_none_0 (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") (block_block_none (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%2") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (- "%arg6" 1) "Symbol4%3") (arith_addf_f64_0 (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%2") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (- "%arg6" 1) "Symbol4%3") "Symbol4%4") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (+ "%arg6" 1) "Symbol4%5") (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%2") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (- "%arg6" 1) "Symbol4%3") "Symbol4%4") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (+ "%arg6" 1) "Symbol4%5") "Symbol4%6") (affine_load_f64_2 "%arg2" (+ "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%7") (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%2") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (- "%arg6" 1) "Symbol4%3") "Symbol4%4") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (+ "%arg6" 1) "Symbol4%5") "Symbol4%6") (affine_load_f64_2 "%arg2" (+ "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%7") "Symbol4%8") (affine_load_f64_2 "%arg2" (- "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%9") (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%2") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (- "%arg6" 1) "Symbol4%3") "Symbol4%4") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (+ "%arg6" 1) "Symbol4%5") "Symbol4%6") (affine_load_f64_2 "%arg2" (+ "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%7") "Symbol4%8") (affine_load_f64_2 "%arg2" (- "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%9") "Symbol4%10") (arith_mulf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%2") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (- "%arg6" 1) "Symbol4%3") "Symbol4%4") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (+ "%arg6" 1) "Symbol4%5") "Symbol4%6") (affine_load_f64_2 "%arg2" (+ "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%7") "Symbol4%8") (affine_load_f64_2 "%arg2" (- "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%9") "Symbol4%10") (arith_constant_f64_0 2.000000e-01 "Symbol4%cst") "Symbol4%11") (affine_store_f64_2 (arith_mulf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%2") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (- "%arg6" 1) "Symbol4%3") "Symbol4%4") (affine_load_f64_2 "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (+ "%arg6" 1) "Symbol4%5") "Symbol4%6") (affine_load_f64_2 "%arg2" (+ "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%7") "Symbol4%8") (affine_load_f64_2 "%arg2" (- "%arg5" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%9") "Symbol4%10") (arith_constant_f64_0 2.000000e-01 "Symbol4%cst") "Symbol4%11") "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg5") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg6") "Symbol4%Pseudo3"))))) (affine_forcontrol_none_0 (affine_forvalue_none_0 1 (- "%0" 1) (* 2 1) "Symbol4%arg7") (block_block_none (affine_forcontrol_none_0 (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") (block_block_none (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%12") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (- "%arg8" 1) "Symbol4%13") (arith_addf_f64_0 (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%12") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (- "%arg8" 1) "Symbol4%13") "Symbol4%14") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (+ "%arg8" 1) "Symbol4%15") (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%12") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (- "%arg8" 1) "Symbol4%13") "Symbol4%14") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (+ "%arg8" 1) "Symbol4%15") "Symbol4%16") (affine_load_f64_2 "%arg3" (+ "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%17") (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%12") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (- "%arg8" 1) "Symbol4%13") "Symbol4%14") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (+ "%arg8" 1) "Symbol4%15") "Symbol4%16") (affine_load_f64_2 "%arg3" (+ "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%17") "Symbol4%18") (affine_load_f64_2 "%arg3" (- "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%19") (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%12") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (- "%arg8" 1) "Symbol4%13") "Symbol4%14") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (+ "%arg8" 1) "Symbol4%15") "Symbol4%16") (affine_load_f64_2 "%arg3" (+ "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%17") "Symbol4%18") (affine_load_f64_2 "%arg3" (- "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%19") "Symbol4%20") (arith_mulf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%12") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (- "%arg8" 1) "Symbol4%13") "Symbol4%14") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (+ "%arg8" 1) "Symbol4%15") "Symbol4%16") (affine_load_f64_2 "%arg3" (+ "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%17") "Symbol4%18") (affine_load_f64_2 "%arg3" (- "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%19") "Symbol4%20") (arith_constant_f64_0 2.000000e-01 "Symbol4%cst") "Symbol4%21") (affine_store_f64_2 (arith_mulf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%12") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (- "%arg8" 1) "Symbol4%13") "Symbol4%14") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (+ "%arg8" 1) "Symbol4%15") "Symbol4%16") (affine_load_f64_2 "%arg3" (+ "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%17") "Symbol4%18") (affine_load_f64_2 "%arg3" (- "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%19") "Symbol4%20") (arith_constant_f64_0 2.000000e-01 "Symbol4%cst") "Symbol4%21") "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%Pseudo6"))) (affine_apply_none (affine_forvalue_none_0 1 (- "%0" 1) (* 2 1) "Symbol4%arg7") "{d0}->{d0+1}") (affine_forcontrol_none_0 (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") (block_block_none (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%12") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (- "%arg8" 1) "Symbol4%13") (arith_addf_f64_0 (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%12") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (- "%arg8" 1) "Symbol4%13") "Symbol4%14") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (+ "%arg8" 1) "Symbol4%15") (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%12") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (- "%arg8" 1) "Symbol4%13") "Symbol4%14") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (+ "%arg8" 1) "Symbol4%15") "Symbol4%16") (affine_load_f64_2 "%arg3" (+ "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%17") (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%12") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (- "%arg8" 1) "Symbol4%13") "Symbol4%14") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (+ "%arg8" 1) "Symbol4%15") "Symbol4%16") (affine_load_f64_2 "%arg3" (+ "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%17") "Symbol4%18") (affine_load_f64_2 "%arg3" (- "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%19") (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%12") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (- "%arg8" 1) "Symbol4%13") "Symbol4%14") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (+ "%arg8" 1) "Symbol4%15") "Symbol4%16") (affine_load_f64_2 "%arg3" (+ "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%17") "Symbol4%18") (affine_load_f64_2 "%arg3" (- "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%19") "Symbol4%20") (arith_mulf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%12") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (- "%arg8" 1) "Symbol4%13") "Symbol4%14") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (+ "%arg8" 1) "Symbol4%15") "Symbol4%16") (affine_load_f64_2 "%arg3" (+ "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%17") "Symbol4%18") (affine_load_f64_2 "%arg3" (- "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%19") "Symbol4%20") (arith_constant_f64_0 2.000000e-01 "Symbol4%cst") "Symbol4%21") (affine_store_f64_2 (arith_mulf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (arith_addf_f64_0 (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%12") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (- "%arg8" 1) "Symbol4%13") "Symbol4%14") (affine_load_f64_2 "%arg3" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (+ "%arg8" 1) "Symbol4%15") "Symbol4%16") (affine_load_f64_2 "%arg3" (+ "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%17") "Symbol4%18") (affine_load_f64_2 "%arg3" (- "%arg7" 1) (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%19") "Symbol4%20") (arith_constant_f64_0 2.000000e-01 "Symbol4%cst") "Symbol4%21") "%arg2" (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg7") (affine_forvalue_none_0 1 (- "%0" 1) 1 "Symbol4%arg8") "Symbol4%Pseudo6")))))))))
    )";
const std::string orginal_filepath = "mlir_output/jacobi_2d_update.mlir"; 


struct Param {
    std::unordered_map<std::string, std::string> param;
    int arg;
    int output;
    int non_return;
    int constant;
    // Constructor (optional, but useful for initialization)
    Param(std::unordered_map<std::string, std::string> s, int arg, int output, int non_return, int constant) : arg(arg), output(output), non_return(non_return), constant(constant) {
        param=s;
    }

    // Default constructor
    Param() : arg(0), output(0), non_return(0), constant(0) {}
};

//map from Egg string to MLIR dialect and operation
std::unordered_map<std::string,std::string> dialect_map ={{"affine","affine"}, {"arith","arith"}};
std::unordered_map<std::string, std::unordered_map<std::string, std::string>> op_map = {
    {"arith", std::unordered_map<std::string, std::string>{{"indexcast", "index_cast"},{"mulf", "mulf"},{"addf", "addf"},{"muli", "muli"},{"addi", "addi"},{"divf", "divf"},{"subf", "subf"},{"divi", "divi"},{"subi", "subi"},{"constant", "constant"}}},
    {"affine", std::unordered_map<std::string, std::string>{{"apply","apply"}, {"load", "load"},{"store", "store"},{"forvalue", "for"},{"forcontrol", "forcontrol"}}},
    {"scf", std::unordered_map<std::string, std::string>{{"forvalue", "for"},{"forcontrol", "forcontrol"}}}
};

int main(){
    OpInfoClass opinfo;
    std::unordered_map<std::string, std::string> param;
    std::string module_information;
    std::unordered_map<std::string, std::string> expr_map;
    std::vector<std::string> processed_expr;
    std::unordered_map<std::string, std::string> output_map;
    std::vector<std::string> unused_expr;
    std::string func_name;
    std::string attributes;
    std::string mlir_string;
    std::unordered_map<std::string,std::string> map;

    // Parameter information
    {
        std::ifstream inFile = utilities::openFile(orginal_filepath);
        std::string line;
        int count_func = 0;
        std::string foundLine_func;
        int count_module = 0;
        std::string foundLine_module;
        while (std::getline(inFile, line)) {
            if (line.find("func.func") != std::string::npos) {
                count_func++;
                foundLine_func = line;
                if (count_func > 1) {
                    throw std::runtime_error("Error - cannot retrieve input parameter");
                }
            }
            else if (line.find("module attributes") != std::string::npos){
                count_module++;
                foundLine_module = line;
                if (count_module > 1) {
                    throw std::runtime_error("Error - cannot retrieve module information");
                }
            }
            else if (line.find("affine_map")!= std::string::npos){
                assert(std::count(line.begin(), line.end(), '>')==2 && line.back()=='>');
                auto output=utilities::split(line,"=<");
                assert(output.size()==3);
                assert(output.at(1)==" affine_map");
                // map[output.at(0)]=output.at(2);
                map[utilities::trim(output.at(0))]=line;
            }
        }
        if (count_func == 1) {
            if (utilities::checkForOneBalancedPair(foundLine_func,0)){
                func_name=utilities::split(foundLine_func," ()").at(1);
                auto pairs = utilities::split(utilities::getContentInFirstPair(foundLine_func,0),",");
                attributes=utilities::getContentInFirstPair(foundLine_func,2);
                for (auto i: pairs){
                    auto param_pair=utilities::split(i,":");
                    if (param_pair.size()==2){
                        param[utilities::trim(param_pair.at(0))] = utilities::trim(param_pair.at(1));
                    }
                    else{
                        throw std::runtime_error("Error parameter: "+i);
                    }
                }
            }
        } else {
            throw std::runtime_error("Error - cannot retrieve input parameter");
        }
        if (count_module == 1) {
            module_information=foundLine_module;
        } else {
            throw std::runtime_error("Error - cannot retrieve module information");
        }
    }

    Param param_info(param,param.size(),0,0,0);
    auto parenthesisPairs = utilities::findParenthesisPairs(inputText,"()");
    std::vector<std::string> stringPairs;
    std::vector<std::string> uniqueVec;
    std::vector<int> indexMap;
    try {
        for (const auto &pair : parenthesisPairs) {
            stringPairs.push_back(inputText.substr(pair.first, pair.second - pair.first+1));
        }
        auto pair = utilities::Set_Index(stringPairs);

        uniqueVec = std::move(pair.first);indexMap = std::move(pair.second);
        std::reverse(uniqueVec.begin(), uniqueVec.end());
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: cannot retrieve expressions." << e.what() << std::endl;
    }
    while (!uniqueVec.empty()){
        auto expression=uniqueVec.back();
        std::cout<<"expression is :"<<expression<<std::endl;

        // Split given expressions into parts based on parentheses and spaces. Then trim each part
        // It then trims each part: if a part is enclosed in double quotes, these quotes are removed.
        std::vector<std::string> split_expression;
        for (const auto& i : utilities::split(expression.substr(1, expression.length() - 2), " ")) {
            split_expression.push_back(
                (i.front() == '"' && i.back() == '"') ? i.substr(1, i.size() - 2) : i
            );
        }
        
        // Return value
        std::string output;
        // If this expression contains a return value
        bool return_flag;

        // Handling the math expression, e.g., (* 2 1).
        if (utilities::isOperator(split_expression[0])){
            bool flag=true;
            for (auto i:split_expression){
                if (i != "+" && i != "-" && i != "*" && i != "/" && !utilities::canBeParsedAsType(i,"f64"))
                        flag=false;
            }
            if (flag){
                output = utilities::evaluateMathExpr(split_expression);
                output_map[output]=expression;
                return_flag=true;
            }
            else{
                std::string infix_expression = utilities::prefixToInfix(expression);
                // Handling the case that can find the corresponding map
                try{
                    auto matchingMap = utilities::findCorrespondingMap(map, infix_expression);
                    // std::cout << "Matching map:" << std::get<0>(matchingMap) << std::endl;
                    // std::cout << "Arguments ():|";
                    // for (const auto &arg : std::get<1>(matchingMap)) {
                    //     std::cout << arg << " ";
                    // }
                    // std::cout << std::endl;
                    // std::cout << "Arguments []:";
                    // for (const auto &arg : std::get<2>(matchingMap)) {
                    //     std::cout << arg << "|";
                    // }
                    // std::cout << std::endl;
                    std::string s="";
                    s=s+std::get<0>(matchingMap)+"(";
                    for (int i=0; i<std::get<1>(matchingMap).size(); i++){
                        if (i!=std::get<1>(matchingMap).size()-1){
                            s=s+std::get<1>(matchingMap).at(i)+" ";
                        }
                        else{
                            s=s+std::get<1>(matchingMap).at(i);
                        }
                    }
                    s=s+")[";
                    for (int i=0; i<std::get<2>(matchingMap).size(); i++){
                        if (i!=std::get<2>(matchingMap).size()-1){
                            s=s+std::get<2>(matchingMap).at(i)+" ";
                        }
                        else{
                            s=s+std::get<2>(matchingMap).at(i);
                        }
                    }
                    s=s+"]";
                    output=s;
                    output_map[output]=expression;
                    return_flag=true;
                    // std::cout << s << std::endl;
                // if cannot find map, direct print it
                } catch (const std::exception &e) {
                    std::string s="";
                    s=s+split_expression.at(1)+split_expression.at(0)+split_expression.at(2);
                    // std::cout<<s<<std::endl;
                    output=s;
                    output_map[output]=expression;
                    return_flag=true;
                }
            }
        }
        // Handling the normal expression, e.g., (arith_indexcast_i32index_0 "%arg0").
        else if (utilities::checkPair(expression,0)){
            auto op_info = utilities::split(split_expression[0],"_");
            // utilities::print(op_info);
            std::string dtype,dialect,op;  std::function<std::vector<int>(int)> pos_func; int dimension; std::vector<int> pos;std::string return_dtype;
            std::function<std::string(std::vector<int>,std::unordered_map<std::string, std::string>, std::vector<std::string>, std::string)> return_func;
            std::string s = "";
            // std::cout<<"expression is :"<<expression<<std::endl;
            // Dialect and operation information
            if (op_info[1]!="forvalue" && op_info[1]!="forcontrol" && op_info[1]!="block" && op_info[1]!="func" && op_info[1]!="apply"){
                try {
                    dialect = dialect_map.at(op_info[0]);
                    op = op_map.at(op_info[0]).at(op_info[1]);
                    return_flag = opinfo.getIfReturn(op_info[0], op_info[1]);
                    pos_func = opinfo.getPosFunc(op_info[0], op_info[1]);
                }
                catch (const std::exception& e) {
                    std::cerr << "Error encountered with op_info[0]: " << op_info[0]
                            << " and op_info[1]: " << op_info[1] << " - " << e.what() << '\n';
                    throw;
                }
                // Adding operation based on output map size
                if (return_flag){
                    // std::cout<<"op:"<<op<<std::endl;
                    return_func=opinfo.getReturnFunc(op_info[0],op_info[1]);
                    if (op=="constant"){
                        output = "%cst_" + std::to_string(param_info.constant);
                    }
                    else{
                        output = "%" + std::to_string(param_info.output);
                    }
                    s = s + output + " = ";
                }
                else{
                    output = "%non_return" + std::to_string(param_info.non_return);
                }
                // Adding dialect and operation
                s = s + dialect + "." + op + " ";
                dimension = std::stoi(op_info[3]);
                //pos is the position of leading variable, for example, %0[%1, %2], %0 is the leading character
                pos = pos_func(dimension);
                std::vector<std::string> sliced;
                std::copy(split_expression.begin() + 1, split_expression.end()-1, std::back_inserter(sliced));
                std::string param_list;
                // std::cout<<"param_list"<<std::endl;
                // utilities::print(sliced);
                std::vector<std::string> operators = {"+", "-", "*", "/"};
                for (int i=0;i<sliced.size();i++){
                    for (const auto& op : operators) {
                        auto exp = utilities::split(sliced.at(i), op);
                        if (exp.size() == 2 and !utilities::canBeParsedAsType(sliced.at(i),"f64")) {
                            sliced.at(i) = exp[0] + " " + op + " " + exp[1];
                        }
                    }
                }
                if (op=="store"){
                    param_list = utilities::convertParam(sliced, {0,1});
                }
                else{
                    param_list = utilities::convertParam(sliced, pos);
                }
                if (return_flag){
                    return_dtype=return_func(pos,param_info.param,sliced,op_info[2]);
                }
                param_info.param[output]=return_dtype;
                s = s + param_list + " : ";
                if (param_info.param.find(sliced.at(pos[0])) != param_info.param.end()){
                    dtype = param_info.param.at(sliced.at(pos[0]));
                }
                else {
                    for (auto j: {"f32", "f64", "i32", "i64"}) {
                        if (utilities::canBeParsedAsType(sliced.at(pos[0]),j)){
                            param_info.param[sliced.at(pos[0])]=j;
                            dtype=j;
                        }
                    }
                }
                for (auto i:pos){
                    if (param_info.param.find(sliced.at(pos[i])) != param_info.param.end()){
                        assert(param_info.param.at(sliced.at(i))==dtype);
                    }
                    else {
                        for (auto j: {"f32", "f64", "i32", "i64"}) {
                            if (utilities::canBeParsedAsType(sliced.at(pos[i]),j)){
                                param_info.param[sliced.at(pos[0])]=j;
                                dtype=j;
                            }
                        }
                    }
                }
                s = s + dtype;
                if (op=="index_cast"){
                    s = s + " to index";
                }
                output_map[output]=expression;
                expr_map[expression]=s;
                if (return_flag){
                    if (op=="constant"){
                        param_info.constant+=1;
                    }
                    else{
                        param_info.output+=1;
                    }
                }
                else{
                    param_info.non_return+=1;
                }
            }
            else if (op_info[1]=="apply"){
                assert(op_info[0]=="affine");
                dialect = dialect_map.at(op_info[0]);
                op = op_map.at(op_info[0]).at(op_info[1]);
                output = "%" + std::to_string(param_info.output);
                param_info.output+=1;
                s = s + output + " = "  + dialect + "." + op + " ";
                std::vector<std::string> sliced;
                std::copy(split_expression.begin() + 1, split_expression.end(), std::back_inserter(sliced));
                auto match_pattern=utilities::replaceString(utilities::replaceString(sliced.back(),"{","("),"}",")");
                match_pattern=utilities::replaceString(match_pattern,"+"," + ");
                match_pattern=utilities::replaceString(match_pattern,"-"," - ");
                match_pattern=utilities::replaceString(match_pattern,"*"," * ");
                match_pattern=utilities::replaceString(match_pattern,"/"," / ");
                match_pattern=utilities::replaceString(match_pattern,"- >","-> ");
                auto map_key="#map"+std::to_string(map.size());
                // match_pattern=utilities::split(match_pattern,"->")[0]+" -> "+utilities::split(match_pattern,"->")[1];
                std::string map_str = map_key+" = affine_map"+"<"+match_pattern+">";
                auto map_flag=true;
                for (auto i:map){
                    std::cout<<i.second<<std::endl;
                    if (utilities::split(i.second,"=").at(1)==utilities::split(map_str,"=").at(1)){
                        map_flag=false;
                        map_key=i.first;
                    }
                }
                if (map_flag){
                    map[map_key]=map_str;
                }
                s=s+map_key+"("+sliced[0]+")";
                output_map[output]=expression;
                expr_map[expression]=s;
                // utilities::print(s);
            }
            else if(op_info[1]=="forvalue"){
                assert(split_expression.size()==5);
                dialect = dialect_map.at(op_info[0]); op = op_map.at(op_info[0]).at(op_info[1]);
                return_flag=opinfo.getIfReturn(op_info[0],op_info[1]);
                return_func=opinfo.getReturnFunc(op_info[0],op_info[1]);
                s = s + dialect + "." + op+" ";
                output = "%arg" + std::to_string(param_info.arg);
                s = s + output + " = ";
                s = s + split_expression[1] + " to " + split_expression[2];
                if (split_expression[3] != "1"){
                    s = s + " step " + split_expression[3];
                }
                std::vector<std::string> sliced;
                std::copy(split_expression.begin() + 1, split_expression.end(), std::back_inserter(sliced));
                return_dtype=return_func(pos,param_info.param,sliced,op_info[2]);
                param_info.param[output]=return_dtype;
                output_map[output]=expression;
                expr_map[expression]=s;
                param_info.arg+=1;
                // std::cout<<s<<std::endl;
            }
            else if (op_info[1]=="block"){
                output = "%non_return" + std::to_string(param_info.non_return);
                for (int i=1; i<split_expression.size();i++){
                    try{
                        std::string str = expr_map.at(output_map.at(split_expression.at(i)));
                        std::istringstream iss(str);
                        std::string line;
                        std::ostringstream oss;
                        while (std::getline(iss, line)) {
                            oss << "  " << line << "\n";
                        }
                        s += oss.str();
                    }
                    catch(const std::runtime_error& e){
                        throw std::runtime_error("Error - cannot retrieve expr map for "+split_expression.at(i));
                    }
                }
                s.pop_back();
                output_map[output]=expression;
                expr_map[expression]=s;
                param_info.non_return+=1;
            }
            else if (op_info[1]=="forcontrol"){
                // utilities::print(split_expression);
                assert(split_expression.size()==3);
                output = "%non_return" + std::to_string(param_info.non_return);
                s = s + expr_map.at(output_map.at(split_expression.at(1)))+" {\n";
                s = s + expr_map.at(output_map.at(split_expression.at(2)))+"\n";
                // std::cout<<"expression is :"<<expression<<std::endl;
                s = s + "}";
                output_map[output]=expression;
                expr_map[expression]=s;
                param_info.non_return+=1;
            }
            else if (op_info[1]=="func"){
                //   func.func @kernel_gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<?x1100xf64>, %arg6: memref<?x1200xf64>, %arg7: memref<?x1100xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
                output = "%non_return" + std::to_string(param_info.non_return);
                s = s + "func.func "+func_name+"(";
                for (int i=1; i<split_expression.size()-1;i++){
                    s=s+split_expression.at(i)+": "+param_info.param.at(split_expression.at(i))+", ";
                }
                s.erase(s.size() - 2);
                s=s+") attributes {"+attributes+"} {\n";

                std::string str = expr_map.at(output_map.at(split_expression.at(split_expression.size()-1)));
                std::istringstream iss(str);
                std::string line;
                std::ostringstream oss;
                while (std::getline(iss, line)) {
                    oss << line << "\n";
                }
                s += oss.str();
                s = s + "  return\n";
                s = s + "}";
                output_map[output]=expression;
                expr_map[expression]=s;
                param_info.non_return+=1;
            }
        }
        processed_expr.push_back(expression);
        if (uniqueVec.size()==1){
            mlir_string=expr_map[expression];
        }
        uniqueVec.pop_back();
        // std::cout<<"expression is :"<<expression<<std::endl;
            // for (auto& i:uniqueVec){
            //     size_t pos = i.find(expression);
            //     if (pos != std::string::npos && i!=expression){
            //         i.replace(pos,expression.length(),output);
            //     }
            // }
        for (auto& i : uniqueVec) {
            // std::cout<<"expression:"<<expression<<"  i:"<<i<<std::endl;
            size_t pos = i.find(expression);
            while (pos != std::string::npos) {
                // Replace the found substring with output
                i.replace(pos, expression.length(), output);

                // Find the next occurrence of the expression
                pos = i.find(expression, pos + output.length());
            }
        }
        // for (auto i:uniqueVec){
        //     utilities::print(i);
        // }
        // std::cout<<"output is :"<<output<<std::endl;
        // std::cout<<"mlir code is :"<<expr_map[expression]<<std::endl;
        // std::cout<<"----------------------"<<std::endl;
    }
    std::string mlir_string_w_module="";
    for (auto i:map){
        mlir_string_w_module=mlir_string_w_module+i.second+"\n";
        // std::cout<<"final mlir code is :"<<std::endl<<mlir_string_w_module<<std::endl;
    }
    mlir_string_w_module=mlir_string_w_module+module_information+"\n";
    std::istringstream iss(mlir_string);
    std::string line;
    std::ostringstream oss;
    while (std::getline(iss, line)) {
        oss << "  " << line << "\n";
    }
    mlir_string_w_module += oss.str();
    mlir_string_w_module +="}";
    std::cout<<"final mlir code is :"<<std::endl<<mlir_string_w_module<<std::endl;
}

        // for (const auto &pair : parenthesisPairs) {
        //     if (utilities::checkForOneBalancedPair(inputText.substr(pair.first, pair.second - pair.first))){

        //     }
        // }