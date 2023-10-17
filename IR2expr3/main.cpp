#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cassert>
#include <vector>
#include <tuple>
#include <functional>
#include "parser.hpp"
#include "utilities.hpp"
#include "graph.hpp"
#include <regex>




std::tuple<std::string, std::string, std::string>
splitPrimary(const std::string& expression) {
    auto parts = utilities::split(expression, "=");
    if (parts.size() != 1 && parts.size() != 2) {
        std::cerr << "Error: Invalid MLIR expression format." << std::endl;
        throw std::runtime_error("Invalid MLIR format1");
    }
    
    std::string outputValue = (parts.size() == 2) ? utilities::split(parts[0]," ")[0] : "";
    std::string toParse = (parts.size() == 2) ? parts[1] : parts[0];
    
    auto secondary_parts = utilities::split(toParse, ":");
    if (secondary_parts.size() != 2) {
        std::cerr << "Error: Invalid MLIR expression format." << std::endl;
        throw std::runtime_error("Invalid MLIR format2");
    }
    return {outputValue, secondary_parts[0], secondary_parts[1]};
}

std::tuple<std::string, std::string, std::vector<std::string>>
analyzeOperation(const std::string& operation_str) {
    auto operation_parts = utilities::split(operation_str, ". ,[]");
    if (operation_parts.size() < 2) {
        std::cerr << "Error: Invalid MLIR expression format." << std::endl;
        throw std::runtime_error("Invalid MLIR format3");
    }
    std::string dialect = operation_parts[0];
    std::string operation = operation_parts[1];
    std::vector<std::string> parameters(operation_parts.begin() + 2, operation_parts.end());
    return {dialect, operation, parameters};
}

std::tuple<int, std::string>
analyzeType(const std::string& type_str) {
    auto main_parts = utilities::split(type_str.substr(1), "<>");
    if (main_parts.size() != 1 && main_parts.size() != 2) {
        std::cerr << "Error: Invalid MLIR type format." << std::endl;
        throw std::runtime_error("Invalid MLIR format");
    }
    
    if (main_parts.size() == 1) {
        return {0, main_parts[0]};  // Not a tensor type, just return data type
    } else {
        auto type_parts = utilities::split(main_parts[1], "x");
        int dimensions = type_parts.size() - 1;
        std::string dataType = type_parts.back();
        return {dimensions, dataType};
    }
}

std::tuple<std::string, std::string, std::string, std::vector<std::pair<std::string, std::tuple<int, std::string>>>, std::string>
parseMLIR_func(const std::string &mlir_str) {
    auto parts = utilities::split(mlir_str, "{}");
    auto main_parts = utilities::split(parts[0], "().@");

    if (main_parts.size() != 5) {
        std::cerr << "Error: Invalid MLIR type format." << std::endl;
        throw std::runtime_error("Invalid MLIR format");
    }


    std::string dialect = utilities::split(main_parts[0], " ")[0];
    std::string operation = utilities::split(main_parts[1], " ")[0];
    std::string function_name = utilities::split(main_parts[2], " ")[0];
    std::string params_str = main_parts[3];
    std::string attributes = main_parts[4] + "{" + parts[1] + "}";

    auto params_list = utilities::split(params_str, ",");
    std::vector<std::pair<std::string, std::tuple<int, std::string>>> params;

    for (const auto &param_str : params_list) {
        auto param_parts = utilities::split(param_str, ":");
        if (param_parts.size() != 2) {
            std::cerr << "Error: Invalid parameter format.2" << std::endl;
            throw std::runtime_error("Invalid parameter format");
        }
        std::string param_name = utilities::split(param_parts[0]," ")[0];
        auto param_type = analyzeType(param_parts[1]);
        // utilities::print(param_name);
        // utilities::print(param_parts[1]);
        // utilities::print(param_type);
        // std::cout<<std::endl;
        // std::cout<<std::endl;
        params.emplace_back(param_name, param_type);
    }

    return {dialect, operation, function_name, params, attributes};
}

std::tuple<std::string, std::string, std::string, std::vector<std::string>>
parseMLIR_scf(const std::string &input) {
    std::vector<std::string> splitResult = utilities::split(input, "=");

    if (splitResult.size() != 2) {
        std::cout << "Error: Invalid MLIR type format." << std::endl;
        throw std::runtime_error("Invalid MLIR format");
    }

    std::string returnValue = splitResult[0];
    std::string loopParameter = splitResult[1];

    std::vector<std::string> returnValueSplit = utilities::split(returnValue, ". ");
    if (returnValueSplit.size() != 3) {
        std::cout << "Error: Invalid MLIR type format." << std::endl;
        throw std::runtime_error("Invalid MLIR format");
    }

    std::string dialect = returnValueSplit[0];
    std::string operation = returnValueSplit[1];
    std::string retValue = returnValueSplit[2];

    std::vector<std::string> loopParameterSplit = utilities::split(loopParameter, " ");
    std::vector<std::string> selectedLoopParameters;
    for (size_t i = 0; i < loopParameterSplit.size(); i += 2) {
        selectedLoopParameters.push_back(loopParameterSplit[i]);
    }

    return std::make_tuple(dialect, operation, retValue, selectedLoopParameters);
}


MyTupleType1 analyzeMLIR(const std::string& expression,
            std::function<std::tuple<std::string, std::string, std::string>(const std::string&)> splitPrimaryFn = splitPrimary,
            std::function<std::tuple<std::string, std::string, std::vector<std::string>>(const std::string&)> analyzeOperationFn = analyzeOperation,
            std::function<std::tuple<int, std::string>(const std::string&)> analyzeTypeFn = analyzeType) {
    auto [output, operation_str, type_str] = splitPrimaryFn(expression);
    auto [dialect, operation, parameters] = analyzeOperationFn(operation_str);
    auto [dimensions, dataType] = analyzeTypeFn(type_str);
    return {output, dialect, operation, parameters, dimensions, dataType};
}

bool checkPattern(const std::string& str) {
    std::regex pattern(R"(^\s*}$|^\s*$)");  // Regular expression for zero or more spaces followed by a closing brace
    return std::regex_match(str, pattern);
}

std::pair<int, int> findBracePair(const std::string &filename, int m, int n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {-1, -1};  // error indicator
    }

    std::string line;
    std::stack<int> bracePositions;
    int firstBraceLine = -1;
    int lastBraceLine = -1;
    int lineNumber = 0;

    while (std::getline(file, line)) {
        ++lineNumber;
        if (lineNumber < m) continue;  // skip lines before the start of the search range
        if (lineNumber > n) break;     // stop reading after the end of the search range

        size_t foundOpen = line.find("{");
        size_t foundClose = line.find("}");
        if (foundOpen != std::string::npos) {
            if (firstBraceLine == -1) firstBraceLine = lineNumber;
            bracePositions.push(lineNumber);
        }
        if (foundClose != std::string::npos) {
            if (!bracePositions.empty()) {
                bracePositions.pop();
                if (bracePositions.empty()) {
                    lastBraceLine = lineNumber;
                }
            }
        }
    }

    file.close();

    if (bracePositions.empty()) {
        return {firstBraceLine, lastBraceLine};
    } else {
        std::cerr << "Error: Unmatched braces" << std::endl;
        return {-1, -1};  // error indicator for unmatched braces
    }
}



int main() {
    Graph empty_graph; Boost_Graph g(empty_graph);
    Vertex unknown_source ("Source","SS","Source","None",{}, {},0);
    g.add_or_update_vertex(&unknown_source);
    Vertex unknown_sink ("Sink","SS","Sink","None",{}, {},0);
    g.add_or_update_vertex(&unknown_sink);
    std::string filepath = "mlir_source/gemm_new.mlir"; // specify your file path here
    int file_length=utilities::countLines(filepath);
    std::ifstream inFile(filepath);
    if (!inFile) {
        std::cerr << "Unable to open file: " << filepath << std::endl;
        std::cerr << "Reason: " << strerror(errno) << std::endl;
        return 1;  // Exit with an error code
    }

    // %3 = memref.load %arg2[%arg7, %arg9] : memref<?x?xi32>
    std::string line;
    int line_num=1;
    std::pair<int,int> brackets={0,file_length};
    std::unordered_map<Vertex*,std::pair<int,int>> Brace;
    while (std::getline(inFile, line)) {
        if (!line.empty()) {
            if (line.find("func")!= std::string::npos){
                MyTupleType2 tuple_output = parseMLIR_func(line);
                std::pair<int,int> range = findBracePair(filepath, brackets.first, brackets.second);
                Func func;
                Vertex* func_op = func.parse(line,tuple_output,g);
                Brace[func_op]=range;
            }
            else if (line.find("affine")!= std::string::npos){
            }
            else if (line.find("scf")!= std::string::npos){
                MyTupleType3 tuple_output = parseMLIR_scf(line);
                std::pair<int,int> range = findBracePair(filepath, brackets.first, brackets.second);
                Scf scf;
                Vertex* scf_op = scf.parse(line,tuple_output,g);
                Brace[scf_op]=range;
            }
            else if (!checkPattern(line)){
                MyTupleType1 tuple_output = analyzeMLIR(line, splitPrimary, analyzeOperation, analyzeType);            // utilities::print(tuple_output);
                if (std::get<1>(tuple_output)=="arith") {
                    Arith arith;
                    arith.parse(line,tuple_output,g);
                } else if (std::get<1>(tuple_output)=="memref") {
                //     // utilities::print(result);
                    Memref memref;
                    memref.parse(line,tuple_output,g);
                }
            }
        }
        line_num+=1;
    }

    // g.printGraph();
    std::ofstream file("output.txt");
    if (file.is_open()) {
        g.printGraph(file);
        file.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
    }
    inFile.close();
      return 0;
}