#ifndef UTILITIES_HPP
#define UTILITIES_HPP
#include <regex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <list>
#include <set>
#include <map>
#include <utility>
#include <typeinfo>

namespace utilities {

inline std::vector<std::string> split(const std::string &s, const std::string& delimiters) {
    std::vector<std::string> tokens;
    size_t prev = 0;
    size_t pos = s.find_first_of(delimiters, prev);
    while (pos != std::string::npos) {
        if (pos > prev) {
            tokens.push_back(s.substr(prev, pos - prev));
        }
        prev = pos + 1;
        pos = s.find_first_of(delimiters, prev);
    }
    if (prev < s.length()) {
        tokens.push_back(s.substr(prev, std::string::npos));
    }
    return tokens;
}

template <typename T>
inline void print(const T&) {
    std::cout << "Unknown Type" << std::endl;
}

inline void print(const int& value) {
    std::cout << "int:" << value << std::endl;
}

inline void print(const size_t& value) {
    std::cout << "size_t:" << value << std::endl;
}


inline void print(const char& value) {
    std::cout << "char:" << value << std::endl;
}

inline void print(const bool& value) {
    std::cout << "bool:" << (value ? "true" : "false") << std::endl;
}

inline void print(const float& value) {
    std::cout << "float:" << value << std::endl;
}

inline void print(const std::string& value) {
    std::cout << "string:" << value << std::endl;
}

template <typename T>
inline void print(const std::vector<T>& vec) {
    std::cout << "vector:";
    for (const auto& item : vec) {
        print(item);
    }
}

template <typename T>
inline void print(const std::list<T>& lst) {
    std::cout << "list:";
    for (const auto& item : lst) {
        print(item);
    }
}

template <typename T, typename U>
inline void print(const std::pair<T, U>& p) {
    std::cout << "pair:";
    print(p.first);
    print(p.second);
}

template <typename T>
inline void print(const std::set<T>& s) {
    std::cout << "set:";
    for (const auto& item : s) {
        print(item);
    }
}

template <typename K, typename V>
inline void print(const std::map<K, V>& m) {
    std::cout << "map:";
    for (const auto& kv : m) {
        print(kv.first);
        print(kv.second);
    }
}

template <typename K, typename V>
inline void print(const std::unordered_map<K, V>& um) {
    std::cout << "unordered_map:";
    for (const auto& kv : um) {
        print(kv.first);
        print(kv.second);
    }
}

template<std::size_t I = 0, typename... Ts>
inline typename std::enable_if<I == sizeof...(Ts), void>::type
printTuple(const std::tuple<Ts...>&) {}

template<std::size_t I = 0, typename... Ts>
inline typename std::enable_if<I < sizeof...(Ts), void>::type
printTuple(const std::tuple<Ts...>& tpl) {
    print(std::get<I>(tpl));
    printTuple<I + 1, Ts...>(tpl);
}

template<typename... Ts>
inline void print(const std::tuple<Ts...>& tpl) {
    std::cout << "tuple:";
    printTuple(tpl);
}



template <typename K, typename V>
std::vector<K> mapKeys(const std::unordered_map<K, V>& m) {
    std::vector<K> keys;
    for (const auto& kv : m) {
        keys.push_back(kv.first);
    }
    return keys;
}

std::string vectorToString(const std::vector<std::string>& vec) {
    std::string result="{";
    for (const auto& str : vec) {
        result += str + std::string(", ");
    }
    result.pop_back();
    result.pop_back();
    result += std::string("}");
    return result;
}

std::string findAndReturnValue(const std::string &target, const std::map<std::string, std::string> &dataMap) {
    std::string foundKey;
    
    for (const auto &pair : dataMap) {
        if (target.find(pair.first) != std::string::npos) {
            assert(foundKey.empty()); // Ensure that we've found only one key in the string
            foundKey = pair.first;
        }
    }
    
    assert(!foundKey.empty()); // Ensure that at least one key was found
    return dataMap.at(foundKey);
}


std::string findAndReturnValue(const std::string &target, const std::unordered_map<std::string, std::string> &dataUnorderedMap) {
// std::unordered_map<std::string, std::string> data = {
//     {"apple", "fruit"},
//     {"carrot", "vegetable"},
//     {"chicken", "meat"},
//     {"rice", "grain"}
// };
// std::string target = "I love eating apple pie.";
// return fruit
    std::string foundKey;
    
    for (const auto &pair : dataUnorderedMap) {
        if (target.find(pair.first) != std::string::npos) {
            // assert(foundKey.empty()); // Ensure that we've found only one key in the string
            foundKey = pair.first;
        }
    }
    
    assert(!foundKey.empty()); // Ensure that at least one key was found
    return dataUnorderedMap.at(foundKey);
}



std::pair<std::string, std::string> findPositionOfOne(std::unordered_map<std::string, std::unordered_map<std::string, int>> _op_type) {

    int sum = 0;
    std::pair<std::string, std::string> position;

    for (const auto& outer_pair : _op_type) {
        for (const auto& inner_pair : outer_pair.second) {
            sum += inner_pair.second;

            if (inner_pair.second == 1) {
                position = {outer_pair.first, inner_pair.first};
            }
        }
    }

    if (sum != 1) {
        throw std::runtime_error("The total sum is not 1");
    }

    return position;
}

template <std::size_t N, std::size_t... Is, typename... Ts>
auto last_n_from_tuple_impl(const std::tuple<Ts...>& tuple, std::index_sequence<Is...>)
{
    return std::get<sizeof...(Ts) - N>(tuple);
}

template <std::size_t N, typename... Ts>
auto last_n_from_tuple(const std::tuple<Ts...>& tuple)
{
    static_assert(N <= sizeof...(Ts), "N must be less than or equal to the tuple size");
    return last_n_from_tuple_impl<N>(tuple, std::make_index_sequence<N>());
}

template<typename T>
auto last_n_from_vector(const std::vector<T>& vec, size_t k) {
    if (k == 0 || k > vec.size()) {
        throw std::out_of_range("k is out of bounds");
    }
    return vec[vec.size() - k];
}

template <typename T>
bool hasIntersection(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    for (const auto &item : vec1) {
        if (std::find(vec2.begin(), vec2.end(), item) != vec2.end()) {
            return true;
        }
    }
    return false;
}

template <typename T>
int countMatchingKeys(const std::map<std::string, T>& name2Edge, const std::string& baseName) {
    std::regex pattern(baseName + "_\\d+");  // The pattern matches baseName followed by an underscore and one or more digits

    int count = 0;
    for (const auto& pair : name2Edge) {
        if (std::regex_match(pair.first, pattern)) {
            count++;
        }
    }
    
    return count;
}


std::string findPositionOfOne(const std::unordered_map<std::string, int>& dataMap) {

    int sum = 0;
    std::string position;

    for (const auto& pair : dataMap) {
        sum += pair.second;

        if (pair.second == 1) {
            position = pair.first;
        }
    }

    if (sum != 1) {
        throw std::runtime_error("The total sum is not 1");
    }

    return position;
}

int countLines(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return -1;
    }

    int lineCount = 0;
    std::string line;
    while (std::getline(file, line)) {
        ++lineCount;
    }

    file.close();
    return lineCount;
}


bool matchPattern(const std::string& line, const std::string& key, const std::string& operation) {
    std::string pattern = key + "." + operation;
    return line.find(pattern) != std::string::npos;
}

std::pair<std::string, int> findPatternInLine(const std::string& line, const std::map<std::string, std::vector<std::pair<std::string, int>>>& dialectDict) {
    int matchCount = 0;
    std::vector<std::string> allMatchedPatterns;
    int operationId = -1; // Placeholder for the operation id

    for (const auto& pair : dialectDict) {
        for (const auto& operation : pair.second) {
            if (matchPattern(line, pair.first, operation.first)) {
                matchCount++;
                allMatchedPatterns.push_back(pair.first+"."+operation.first);
                operationId = operation.second;
            }
        }
    }

    if (matchCount != 1) {
        std::string errorStr = "Matched pattern is not 1 for line: " + line + ". Patterns found: ";
        for (const auto& pat : allMatchedPatterns) {
            errorStr += pat + ", ";
        }
        errorStr = errorStr.substr(0, errorStr.length() - 2); // remove the trailing ", "
        throw std::runtime_error(errorStr);
    }


    return {allMatchedPatterns.front(),operationId}; // Return the matched pattern
}

}  // namespace utilities
#endif