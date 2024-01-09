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
#include <cerrno>
#include <typeinfo>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <unordered_set>

namespace utilities {

/* 
Function: split
Purpose: Splits a string into a vector of substrings based on specified delimiters.
Parameters:
   - s: The string to be split.
   - delimiters: A string containing all delimiter characters.
Returns: A std::vector of strings, each being a substring of 's' split by any of the delimiter characters.
Example:
std::string str = "hello, world welcome to,split function";
std::string delims = " ,";
std::vector<std::string> splitStr = split(str, delims);
'splitStr' will now contain {"hello", "world", "welcome", "to", "split", "function"}
*/
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


// Print function for various datatype
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

/* 
Function: mapKeys
Purpose: Extracts the keys from a given std::unordered_map and stores them in a std::vector.
Parameters:
   - m: A constant reference to an std::unordered_map of key-value pairs.
Returns: A std::vector containing all the keys from the map.
Usage Example:
std::unordered_map<std::string, int> map = {
    {"apple", 1},
    {"banana", 2},
    {"cherry", 3}
};
std::vector<std::string> keys = mapKeys(map);
'keys' will now contain {"apple", "banana", "cherry"}
*/
template <typename K, typename V>
std::vector<K> mapKeys(const std::unordered_map<K, V>& m) {
    std::vector<K> keys;
    for (const auto& kv : m) {
        keys.push_back(kv.first);
    }
    return keys;
}

// Function: vectorToString
// Purpose: Converts a vector of strings into a single string representation.
// Parameters:
//    - vec: A constant reference to a vector of strings.
// Returns: A string that concatenates all the elements of the vector, separated by commas,
//          and enclosed in curly braces.
// Usage Example:
// std::vector<std::string> myVec = {"apple", "banana", "cherry"};
// std::string strRepresentation = vectorToString(myVec);
// 'strRepresentation' will now be "{apple, banana, cherry}"

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


// Function: findAndReturnValue
// Purpose: Searches for a key within a target string and returns its corresponding value from a map.
// Parameters:
//    - target: A string in which to search for keys.
//    - dataMap: A map (std::map) where each key-value pair is a string.
// Returns: The value corresponding to the first key found within the target string.
// Note: The function asserts that exactly one key is found in the target string.
// Usage Example:
// Suppose you have a map and a target string.
// std::map<std::string, std::string> dataMap = {{"apple", "fruit"}, {"car", "vehicle"}};
// std::string target = "I have an apple and a car.";
// You can find the first key from the map that appears in the target string and return its value.
// std::string value = findAndReturnValue(target, dataMap);
// 'value' will now be "fruit" since "apple" is the first key in the target string that matches a key in the map.
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

// Similar function for std::unordered_map
std::string findAndReturnValue(const std::string &target, const std::unordered_map<std::string, std::string> &dataUnorderedMap) {

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


// Function: findPositionOfOne
// Purpose: Finds the position (a pair of strings) where the value '1' occurs in a nested unordered map.
// Parameters:
//    - _op_type: A nested std::unordered_map. The outer map's keys and the inner map's keys are strings, and the inner map's values are integers.
// Returns: A std::pair of strings representing the keys (from the outer and inner maps) where the value '1' is found.
// Throws: std::runtime_error if the total sum of all values in the inner maps is not equal to 1.
// Usage Example:
// Suppose you have a nested unordered map like this:
// std::unordered_map<std::string, std::unordered_map<std::string, int>> opType = {
//     {"Arith", {{"muli", 0}, {"addi", 1}}},
//     {"Affine", {{"load", 0}, {"store", 0}}}
// };
// You can find the position of the value '1' using the findPositionOfOne function.
// std::pair<std::string, std::string> pos = findPositionOfOne(opType);
// 'pos' will now be {"Arith", "addi"}, indicating the position of the value '1'.
// Note this function asserts that exactly one "1" is found in the target nested unordered map.
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
    // Throw an error if the total sum is not equal to 1
    if (sum != 1) {
        throw std::runtime_error("The total sum is not 1");
    }

    return position;
}

// Purpose: Version of 2-depth nested unordered_map
// Usage Example:
// std::unordered_map<std::string, int> map = {{"A", 0}, {"B", 1}, {"C", 0}};
// std::string position = findPositionOfOne(map);
// 'position' will be "B", as it is the only key with the value '1'.
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

// Function: last_n_from_vector
// Purpose: Retrieves the n-th last element from a vector.
// Parameters:
//    - vec: A constant reference to a vector of type T.
//    - k: The position from the end of the vector (1-based index) of the element to retrieve.
// Returns: The n-th last element of the vector.
// Throws: std::out_of_range if k is 0 or greater than the size of the vector.
// Usage Example:
// Suppose you have a vector of integers.
// std::vector<int> myVector = {10, 20, 30, 40, 50};
// You want to get the 2nd last element from this vector.
// int element = last_n_from_vector(myVector, 2);
// 'element' will now be 40, which is the second last element in the vector.
template<typename T>
auto last_n_from_vector(const std::vector<T>& vec, size_t k) {
    if (k == 0 || k > vec.size()) {
        throw std::out_of_range("k is out of bounds");
    }
    return vec[vec.size() - k];
}

// Function: last_n_from_tuple
// Purpose: Extracts the N-th last element from a tuple.
// Parameters:
//    - tuple: A constant reference to a tuple.
// Returns: The N-th last element of the tuple.
// Note: Static assertion ensures that N is not greater than the number of elements in the tuple.
// Usage Example:
// Suppose you have a tuple of various types.
// std::tuple<int, std::string, double, char> myTuple = std::make_tuple(1, "hello", 3.14, 'a');
// You want to get the 2nd last element from this tuple.
// auto element = last_n_from_tuple<2>(myTuple);
// 'element' will now be 3.14, which is the second last element in the tuple.
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



// Function: hasIntersection
// Purpose: Determines if there is any common element between two vectors.
// Parameters:
//    - vec1: A vector of type T.
//    - vec2: Another vector of type T.
// Returns: True if there is at least one common element, false otherwise.
// Usage Example:
// std::vector<int> vector1 = {1, 2, 3};
// std::vector<int> vector2 = {3, 4, 5};
// bool intersect = hasIntersection(vector1, vector2);
// We have 'intersect' will be true, since element 3 is common in both vectors.

template <typename T>
bool hasIntersection(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    for (const auto &item : vec1) {
        if (std::find(vec2.begin(), vec2.end(), item) != vec2.end()) {
            return true;
        }
    }
    return false;
}

// Function: countMatchingKeys
// Purpose: Counts the number of keys in a map that match a specific regex pattern.
// Parameters:
//    - name2Edge: A map where keys are strings.
//    - baseName: The base name to form the regex pattern, which will be followed by an underscore and one or more digits.
// Returns: The count of keys that match the pattern.
// Usage Example:
// std::map<std::string, int> map = {{"apple_1", 10}, {"apple_2", 20}, {"banana_1", 30}};
// int count = countMatchingKeys(map, "apple");
// 'count' will be 2, as 'apple_1' and 'apple_2' match the pattern.

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


// Function: countLines
// Purpose: Counts the number of lines in a text file.
// Parameters:
//    - filename: The name of the file to be read.
// Returns: The number of lines in the file, or -1 if the file cannot be opened.
// Notes: This function opens the file, counts lines, and then closes the file.
/* 
Usage Example:
// To count the number of lines in a file named "example.txt"
std::string filename = "example.txt";
int lines = countLines(filename);
// 'lines' will now contain the number of lines in 'example.txt'.
*/
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

// Function: openFile
// Purpose: Opens a file for reading and checks for errors.
// Parameters:
//    - filepath: The path of the file to be opened.
// Returns: A std::ifstream object representing the opened file.
// Throws: std::runtime_error if the file cannot be opened.
// Notes: The function reports the specific error if the file cannot be opened.
std::ifstream openFile(const std::string& filepath) {
    std::ifstream inFile(filepath);
    if (!inFile) {
        std::cerr << "Unable to open file: " << filepath << std::endl;
        std::cerr << "Reason: " << strerror(errno) << std::endl;
        throw std::runtime_error("File open error");
    }
    return inFile;
}

// Function: matchPattern
// Purpose: Determines if a specified pattern exists in a given line.
// Parameters:
//    - line: The string in which to search for the pattern.
//    - key: The first part of the pattern.
//    - operation: The second part of the pattern, which is concatenated to the key.
// Returns: True if the pattern is found in the line, false otherwise.
/* 
Usage Example:
// To check if the pattern "key.operation" exists in a string "line"
std::string line = "This is a test for key.operation in the line";
std::string key = "key";
std::string operation = "operation";
bool isMatch = matchPattern(line, key, operation);
// 'isMatch' will be true as "key.operation" is present in 'line'.
*/
bool matchPattern(const std::string& line, const std::string& key, const std::string& operation) {
    std::string pattern = key + "." + operation;
    return line.find(pattern) != std::string::npos;
}

// Function: findPatternInLine
// Purpose: Finds a specific pattern in a line based on a map of dialects and operations.
// Parameters:
//    - line: The line in which to search for patterns.
//    - dialectDict: A map where keys are dialect names and values are vectors of pairs of operations and their IDs.
// Returns: A pair of the matched pattern and its corresponding operation ID.
// Throws: std::runtime_error if there is not exactly one match found.
/* 
Usage Example:
// To find a pattern in a line using a predefined dictionary of dialects
std::string line = "%5 = affine.load %arg7[%arg9, %arg10] : memref<?x1100xf64>";
std::map<std::string, std::vector<std::pair<std::string, int>>> dialectDict = {
    {"affine", {{"load", 1}, {"store", 2}}},
    {"arith", {{"addi", 1}, {"muli", 2}}}
};
std::pair<std::string, int> result;
try {
    result = findPatternInLine(line, dialectDict);
    // 'result' will contain {"affine.load", 1}
}
Find the operation in the selected MLIR line
*/
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


// Function: findParenthesisPairs
// Purpose: Identifies and returns pairs of indices corresponding to matching opening and closing parentheses in a string.
//          The function ensures that the output pairs are ordered such that a later pair in the list cannot be contained
//          within an earlier pair. This is achieved by using a stack to match each closing parenthesis with the most
//          recent unmatched opening parenthesis.
// Parameters:
//    - text: The string in which to find pairs of matching parentheses.
// Returns: A vector of pairs, where each pair contains the indices of a matching set of parentheses.
// Throws: std::runtime_error if the parentheses in the string are unbalanced.
// Note: The output of pairs has already follow the orders where later pairs cannot be contained in prior pairs
/*
Usage Example:
// To find pairs of matching parentheses in a string
std::string inputText = "a(b)c(d(e)f)g";
std::vector<std::pair<int, int>> parenthesisPairs;
try {
    parenthesisPairs = findParenthesisPairs(inputText);
    // Print each pair of indices
    for (const auto &pair : parenthesisPairs) {
        std::cout << "(" << pair.first << ", " << pair.second << "), ";
    }
} catch (const std::runtime_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
// For the input "a(b)c(d(e)f)g", the output will be "(1, 3), (5, 11), (6, 10), "
*/
std::vector<std::pair<int, int>> findParenthesisPairs(const std::string &text) {
    std::stack<int> stack;
    std::vector<std::pair<int, int>> pairs;

    for (int i = 0; i < text.length(); ++i) {
        if (text[i] == '(') {
            stack.push(i);
        } else if (text[i] == ')') {
            if (stack.empty()) {
                throw std::runtime_error("Unbalanced parentheses: Extra closing parenthesis detected.");
            }
            int openingIndex = stack.top();
            stack.pop();
            pairs.push_back(std::make_pair(openingIndex, i));
        }
    }

    // Check if there are unmatched opening parentheses left in the stack
    if (!stack.empty()) {
        throw std::runtime_error("Unbalanced parentheses: Extra opening parenthesis detected.");
    }

    return pairs;
}

/*
Function: checkForOneBalancedPair
Purpose: Checks if a given string contains exactly one balanced pair of specified parentheses.
Parameters:
   - text: The string in which to check for a balanced pair of parentheses.
   - type: The type of parentheses to check for. 0 for "()", 1 for "{}", 2 for "[]", 3 for "<>".
Returns: bool - true if the string contains exactly one balanced pair of the specified type of parentheses,
        false otherwise.
Example Usage:
   std::string inputText = "(example)";
   bool result = checkForOneBalancedPair(inputText, 0);
   // For the input "(example)" and type 0, result will be true
*/
bool checkForOneBalancedPair(const std::string& text, int type) {
    char openChar, closeChar;

    // Set the type of parentheses to check
    switch (type) {
        case 0: openChar = '('; closeChar = ')'; break;
        case 1: openChar = '{'; closeChar = '}'; break;
        case 2: openChar = '['; closeChar = ']'; break;
        case 3: openChar = '<'; closeChar = '>'; break;
        default: return false; // Invalid type
    }

    int openCount = 0, closeCount = 0;

    for (char ch : text) {
        if (ch == openChar) {
            if (openCount > 0) {
                // More than one opening parenthesis found
                return false;
            }
            openCount++;
        } else if (ch == closeChar) {
            if (closeCount > 0 || openCount == 0) {
                // Unbalanced parentheses or closing parenthesis before opening
                return false;
            }
            closeCount++;
        }
    }
    // True if there's exactly one balanced pair, false otherwise
    return openCount == closeCount;
}

// Function: getContentInFirstPair
// Purpose: Retrieves the content inside the first balanced pair of specified brackets in a string.
// Parameters:
//    - text: The string from which to extract the content.
//    - bracketType: An integer representing the type of brackets. 0 for "()", 1 for "[]", 2 for "{}", 3 for "<>".
// Returns: A string containing the content inside the first balanced pair of specified brackets.
// Throws: std::runtime_error if no balanced pair of specified brackets is found.
// Example Usage:
//    std::string inputText = "Example {extract this} content";
//    std::string content = getContentInFirstPair(inputText, 2);
//    std::cout << "Content inside brackets: " << content << std::endl; // "extract this"

std::string getContentInFirstPair(const std::string& text, int bracketType) {
    std::vector<std::pair<char, char>> brackets = {{'(', ')'}, {'[', ']'}, {'{', '}'}, {'<', '>'}};
    
    if (bracketType < 0 || bracketType >= brackets.size()) {
        throw std::runtime_error("Invalid bracket type");
    }

    auto [openChar, closeChar] = brackets[bracketType];
    std::size_t openPos = text.find(openChar);
    std::size_t closePos = text.find(closeChar, openPos);

    if (openPos == std::string::npos || closePos == std::string::npos) {
        throw std::runtime_error("No balanced pair of specified brackets found");
    }

    return text.substr(openPos + 1, closePos - openPos - 1);
}

// Function: Set_Index
// Purpose: Processes a vector of strings to remove duplicates and create an index mapping.
//          It maintains the order of first occurrence for each unique string and assigns a new index to it.
//          The function then creates a mapping vector where each element's position corresponds to the index 
//          in the original vector and its value corresponds to the new index of the unique string.
// Parameters:
//    - originalVec: A constant reference to a vector of strings to be processed.
// Returns: A pair consisting of two vectors. The first vector contains the unique strings in the order
//          they first appear in the original vector. The second vector contains the index mapping from 
//          the original vector to the new vector of unique strings.
// Note:: uniqueVec preserves the order of the elements as they first appear in the originalVec.
// Example Usage:
//    std::vector<std::string> originalVec = {"apple", "banana", "apple", "orange", "banana"};
//    auto [uniqueVec, indexMap] = Set_Index(originalVec);
//    // uniqueVec will contain {"apple", "banana", "orange"}
//    // indexMap will contain {1, 2, 1, 3, 2}
std::pair<std::vector<std::string>, std::vector<int>> Set_Index(const std::vector<std::string>& originalVec) {
    std::unordered_map<std::string, int> uniqueMap;
    std::vector<std::string> uniqueVec;
    std::vector<int> indexMap;

    int newIndex = 1;
    for (const auto& elem : originalVec) {
        if (uniqueMap.find(elem) == uniqueMap.end()) {
            uniqueMap[elem] = newIndex++;
            uniqueVec.push_back(elem);
        }
        indexMap.push_back(uniqueMap[elem]);
    }

    return {uniqueVec, indexMap};
}

// Function: evaluateMathExpr
// Purpose: Evaluates a mathematical expression given in the form of a string vector. The vector
//          is expected to have three elements: an operator (either "+", "-", "*", or "/") and
//          two operands. The function performs the specified operation on the operands.
//          It handles division by zero and invalid numeric inputs.
// Parameters:
//    - split_expression: A constant reference to a vector of strings representing the operator
//      and operands of the mathematical expression.
// Returns: A string containing the result of the evaluated expression or an error message if the
//          expression is invalid or if a runtime error occurs during evaluation.
// Note: The function assumes that the input vector is properly formatted. Incorrect formatting or
//       non-numeric operand values will result in an error message.
// Example Usage:
//    std::vector<std::string> expression = {"*", "4", "5"};
//    std::string result = evaluateMathExpr(expression);
//    // result will contain "20"
std::string evaluateMathExpr(const std::vector<std::string>& split_expression) {
    if (split_expression.size() != 3) {
        std::string expression = "";
        for (const auto& e : split_expression) {
            expression += e + " ";
        }
        throw std::runtime_error("Invalid expression format: " + expression);
    }

    std::string result;
    try {
        float num1 = std::stof(split_expression[1]);
        float num2 = std::stof(split_expression[2]);
        float num;
        if (split_expression[0] == "+") {
            num = num1 + num2;
        } else if (split_expression[0] == "-") {
            num = num1 - num2;
        } else if (split_expression[0] == "*") {
            num = num1 * num2;
        } else if (split_expression[0] == "/") {
            if (num2 == 0) {
                throw std::runtime_error("");
            }
            num = num1 / num2;
        }
        if (std::floor(num) == num) {
            result = std::to_string(static_cast<int>(num));
        } else {
            result = std::to_string(num);
            result.erase(result.find_last_not_of('0') + 1, std::string::npos);
        }
    } catch(const std::exception& e) {
        std::string expression = "";
        for (const auto& e : split_expression) {
            expression += e + " ";
        }
        throw std::runtime_error("Invalid expression format: " + expression);
    }
    return result;
}

// Checks if a string is one of the specified operators (+, -, *, or /)
bool isOperator(const std::string& ch) {
    return ch.length() == 1 && (ch == "+" || ch == "-" || ch == "*" || ch == "/");
}

// Function: trim
// Purpose: Trims whitespace from both the left and right sides of the string.
// Parameters:
//    - str: The string to trim.
// Returns: A trimmed version of the input string.
// Usage example:
// std::string example = "   Hello, World!   ";
// std::string trimmed = trim(example);
std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) {
        return ""; // The string contains only whitespace characters
    }
    size_t end = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(start, end - start + 1);
}


/* 
Function: transformString
Purpose: Transforms a given string by converting "[" to "<", "]" to ">", spaces inside brackets to "~", and commas inside brackets to "|".
Parameters:
   - input: The string to be transformed.
Returns: A transformed std::string with the specified character replacements.
Example:
   std::string input = "affine.load %arg6[%arg8, %arg9]";
   std::string transformed = transformString(input);
   'transformed' will now be "affine.load %arg6(%arg8|~%arg9)"
*/

std::string transformString(const std::string &input) {
    std::string result;
    bool insideBrackets = false;

    for (char c : input) {
        if (c == '[') {
            insideBrackets = true;
            result += '<';
        } else if (c == ']') {
            insideBrackets = false;
            result += '>';
        } else if (insideBrackets) {
            if (c == ' ') {
                result += '~';
            } else if (c == ',') {
                result += '|';
            } else {
                result += c;
            }
        } else {
            result += c;
        }
    }

    return result;
}

// Function: convertParam
// Purpose: Formats a vector of strings based on specified leading positions. Strings at leading positions are kept separate, while others are grouped in square brackets following the leading string. If two leading strings are adjacent, square brackets are not used.
// Parameters:
//    - strings: A std::vector of strings to be formatted.
//    - leadingPositions: A std::vector of integers representing the positions in 'strings' that are leading.
// Returns: A std::string representing the formatted string sequence.
// Throws: No explicit exceptions thrown.
// Usage Example:
// Suppose you have a vector of strings like this:
// std::vector<std::string> strings = {"a", "b", "c", "d", "e"};
// and a vector of leading positions like this:
// std::vector<int> leadingPositions = {0, 2};
// You can format the strings using the convertParam function:
// std::string result = convertParam(strings, leadingPositions);
// 'result' will now be "a[b], c[d, e]". In this format, 'a' and 'c' are leading strings, and the subsequent strings are grouped accordingly.
// Note that if two leading strings are adjacent (e.g., positions {0, 1}), no brackets are used between them.
std::string convertParam(const std::vector<std::string>& strings, const std::vector<int>& leadingPositions) {
    std::unordered_set<int> leadPosSet(leadingPositions.begin(), leadingPositions.end());
    std::string result;
    
    for (size_t i = 0; i < strings.size(); ++i) {
        if (leadPosSet.count(i)) { // If it's a leading string
            if (!result.empty()) {
                result += ", ";
            }
            result += strings[i];
            if (i + 1 < strings.size() && !leadPosSet.count(i + 1)) {
                result += "[";
            }
        } else { // If it's not a leading string
            result += strings[i];
            if (i + 1 == strings.size() || leadPosSet.count(i + 1)) {
                result += "]";
            } else {
                result += ", ";
            }
        }
    }

    return result;
}

// Deprecated function for print tuple
// template<std::size_t I = 0, typename... Ts>
// inline typename std::enable_if<I == sizeof...(Ts), void>::type
// printTuple(const std::tuple<Ts...>&) {}

// template<std::size_t I = 0, typename... Ts>
// inline typename std::enable_if<I < sizeof...(Ts), void>::type
// printTuple(const std::tuple<Ts...>& tpl) {
//     print(std::get<I>(tpl));
//     printTuple<I + 1, Ts...>(tpl);
// }

// template<typename... Ts>
// inline void print(const std::tuple<Ts...>& tpl) {
//     std::cout << "tuple:";
//     printTuple(tpl);
// }


}  // namespace utilities
#endif