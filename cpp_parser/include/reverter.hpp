#include <iostream>
#include <unordered_map>
#include <functional>
#include <vector>
#include <typeinfo>
#include "utilities.hpp"
#include "graph.hpp"


// First element: a boolean - Flag indicating if this struct includes a return value.
// List of indices denoting the positions of primary parameters in a parameter list.
// For example, in "%arg5[%arg1, %arg2]", %arg5 would be the primary parameter.
// Function pointer to a method that processes inputs and returns a string.
// This method accepts two parameters:
// 1. A vector of strings, representing input strings.
// 2. A vector of unordered maps (each map is from string to string), representing the structure of input parameters.
// struct OpInfo {
//     bool if_return;
//     std::vector<int> pos;
//     std::function<std::string(const std::vector<std::string>&, 
//                               const std::vector<std::unordered_map<std::string, std::string>>&)> func;
//     // Default constructor
//     OpInfo() : if_return(true), pos({}), func([](const std::vector<std::string>&, 
//                                                   const std::vector<std::unordered_map<std::string, std::string>>& mapVec) -> std::string {
//                                                       return "example return";
//                                                   }) {}
//     // Parameterized constructor
//     OpInfo(bool if_return, std::vector<int> pos, 
//            std::function<std::string(const std::vector<std::string>&, 
//                                      const std::vector<std::unordered_map<std::string, std::string>>&)> func) 
//         : if_return(if_return), pos(pos), func(func) {}
// };

// std::string myCustomFunction(const std::vector<std::string>& inputStrings, 
//                              const std::vector<std::unordered_map<std::string, std::string>>& maps) {
//     assert(maps.size()==inputStrings.size()-1);
//     return "Processed by custom function";
// }


class OpInfoClass {
    private:
        std::unordered_map<std::string, std::unordered_map<std::string, bool>> _if_return;
        std::unordered_map<std::string, std::unordered_map<std::string, std::function<std::vector<int>(int)>>> _pos;
        std::unordered_map<std::string, std::unordered_map<std::string, std::function<std::string(std::vector<int>,std::unordered_map<std::string, std::string>, std::vector<std::string>)>>> _return;
        std::vector<int> func1(int dimension) { return {};}
        std::vector<int> func2(int dimension) { return {0,dimension+1}; }
        std::vector<int> func3(int dimension) { return {0};}
        std::vector<int> func4(int dimension) { return {1};}
        std::string func5(std::vector<int> pos, std::unordered_map<std::string, std::string> param, std::vector<std::string> sliced) { return "index";}
        std::string func6(std::vector<int> pos, std::unordered_map<std::string, std::string> param, std::vector<std::string> sliced) { 
            std::string dtype = param.at(sliced.at(pos[0]));
            for (auto i:pos){
                assert(param.at(sliced.at(i))==dtype);
            }
            return dtype;
        }
        std::string func7(std::vector<int> pos, std::unordered_map<std::string, std::string> param, std::vector<std::string> sliced) { 
            std::string dtype = param.at(sliced.at(pos[0]));
            for (auto i:pos){
                assert(param.at(sliced.at(i))==dtype);
            }
            return utilities::split(dtype,"<x>").back();
        }
    public:
        OpInfoClass() {
            _if_return = {
                {"arith", {{"none", false}, {"addi", true}, {"muli", true}, {"addf", true}, {"mulf", true}, {"indexcast", true}}},
                {"memref", {{"none", false}, {"load", true}, {"store", false}}},
                {"scf", {{"none", false}, {"forvalue", true}, {"forcontrol", false}}},
                {"affine", {{"none", false}, {"forvalue", true}, {"forcontrol", false}, {"if", false}, {"load", true}, {"store", false}}},
                {"block", {{"block", false}}}
            };

            _pos["block"]["block"] = [this](int dimension) { return this->func1(dimension); };
            _pos["arith"]["addi"] = [this](int dimension) { return this->func2(dimension); };
            _pos["arith"]["muli"] = [this](int dimension) { return this->func2(dimension); };
            _pos["arith"]["addf"] = [this](int dimension) { return this->func2(dimension); };
            _pos["arith"]["mulf"] = [this](int dimension) { return this->func2(dimension); };
            _pos["arith"]["indexcast"] = [this](int dimension) { return this->func3(dimension); };
            _pos["memref"]["load"] = [this](int dimension) { return this->func3(dimension); };
            _pos["memref"]["store"] = [this](int dimension) { return this->func4(dimension); };
            _pos["affine"]["load"] = [this](int dimension) { return this->func3(dimension); };
            _pos["affine"]["store"] = [this](int dimension) { return this->func4(dimension); };
            _return["scf"]["forvalue"] = [this](std::vector<int> pos,std::unordered_map<std::string, std::string> param, std::vector<std::string> sliced) { return this->func5(pos,param,sliced); };
            _return["affine"]["forvalue"] = [this](std::vector<int> pos,std::unordered_map<std::string, std::string> param, std::vector<std::string> sliced) { return this->func5(pos,param,sliced); };
            _return["arith"]["indexcast"] = [this](std::vector<int> pos,std::unordered_map<std::string, std::string> param, std::vector<std::string> sliced) { return this->func5(pos,param,sliced); };
            _return["arith"]["addi"] = [this](std::vector<int> pos,std::unordered_map<std::string, std::string> param, std::vector<std::string> sliced) { return this->func6(pos,param,sliced); };
            _return["arith"]["addf"] = [this](std::vector<int> pos,std::unordered_map<std::string, std::string> param, std::vector<std::string> sliced) { return this->func6(pos,param,sliced); };
            _return["arith"]["muli"] = [this](std::vector<int> pos,std::unordered_map<std::string, std::string> param, std::vector<std::string> sliced) { return this->func6(pos,param,sliced); };
            _return["arith"]["mulf"] = [this](std::vector<int> pos,std::unordered_map<std::string, std::string> param, std::vector<std::string> sliced) { return this->func6(pos,param,sliced); };
            _return["affine"]["load"] = [this](std::vector<int> pos,std::unordered_map<std::string, std::string> param, std::vector<std::string> sliced) { return this->func7(pos,param,sliced); };
            _return["memref"]["load"] = [this](std::vector<int> pos,std::unordered_map<std::string, std::string> param, std::vector<std::string> sliced) { return this->func7(pos,param,sliced); };
        }
        // Getter for _if_return map
        bool getIfReturn(const std::string& category, const std::string& operation) const {
            auto catIt = _if_return.find(category);
            if (catIt != _if_return.end()) {
                auto opIt = catIt->second.find(operation);
                if (opIt != catIt->second.end()) {
                    return opIt->second;
                }
            }
            throw std::runtime_error("Category or operation not found in _if_return");
        }

        // Getter for a specific _pos function
        std::function<std::vector<int>(int)> getPosFunc(const std::string& category, const std::string& operation) const {
            auto catIt = _pos.find(category);
            if (catIt != _pos.end()) {
                auto opIt = catIt->second.find(operation);
                if (opIt != catIt->second.end()) {
                    return opIt->second;
                }
            }
            throw std::runtime_error("Category or operation not found in _pos");
        }
        // Getter for a specific _return function
        std::function<std::string(std::vector<int>,std::unordered_map<std::string, std::string>, std::vector<std::string>)> getReturnFunc(const std::string& category, const std::string& operation) const {
            auto catIt = _return.find(category);
            if (catIt != _return.end()) {
                auto opIt = catIt->second.find(operation);
                if (opIt != catIt->second.end()) {
                    return opIt->second;
                }
            }
            throw std::runtime_error("Category or operation not found in _return");
        }

        // Optional: Getter for the entire _if_return map
        const std::unordered_map<std::string, std::unordered_map<std::string, bool>>& getIfReturnMap() const {
            return _if_return;
        }

        // Optional: Getter for the entire _pos map
        const std::unordered_map<std::string, std::unordered_map<std::string, std::function<std::vector<int>(int)>>>& getPosMap() const {
            return _pos;
        }
};
