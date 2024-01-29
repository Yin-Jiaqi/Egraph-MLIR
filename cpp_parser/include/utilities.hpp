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
#include <cctype>


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
        std::cout<<"key: ";
        print(kv.first);
        std::cout<<"value: ";
        print(kv.second);
    }
}

/* 
Function: toLowerCase
Purpose: Converts all characters in a given string to their lowercase equivalents.

Parameters:
   - str: The string to be converted to lowercase.

Returns: A new string with all characters converted to lowercase.

Description:
   - The function creates a new string, 'lowerCaseStr', as a copy of the input string 'str'.
   - It then iterates over each character in 'lowerCaseStr'.
   - For each character, it uses the std::tolower function to convert it to lowercase. 
     The character is cast to 'unsigned char' to properly handle characters outside the basic ASCII range.
   - After all characters have been converted, the function returns the modified string.

Example Usage:
    std::string originalStr = "Hello, World!";
    std::string lowerStr = toLowerCase(originalStr);
    std::cout << "Lowercase string: " << lowerStr << std::endl;
    // This would output "Lowercase string: hello, world!".

Note:
   - The function is safe to use with strings containing special characters or non-ASCII characters, 
     as it casts each character to 'unsigned char' before applying std::tolower.
   - The function creates a copy of the input string and does not modify the original string.
*/
std::string toLowerCase(const std::string& str) {
    std::string lowerCaseStr = str;
    for (char &c : lowerCaseStr) {
        c = std::tolower(static_cast<unsigned char>(c));
    }
    return lowerCaseStr;
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
//    - parenthesisType: A string specifying the type of parentheses to check. Should be "()", "[]", "{}", or "<>".
// Returns: A vector of pairs, where each pair contains the indices of a matching set of parentheses.
// Throws: std::runtime_error if the parentheses in the string are unbalanced.
// Note: The output of pairs has already follow the orders where later pairs cannot be contained in prior pairs
/*
Usage Example:
// To find pairs of matching parentheses in a string
std::string inputText = "a(b)c(d(e)f)g";
std::vector<std::pair<int, int>> parenthesisPairs;
try {
    parenthesisPairs = findParenthesisPairs(inputText,"()");
    // Print each pair of indices
    for (const auto &pair : parenthesisPairs) {
        std::cout << "(" << pair.first << ", " << pair.second << "), ";
    }
} catch (const std::runtime_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
// For the input "a(b)c(d(e)f)g", the output will be "(1, 3), (5, 11), (6, 10), "
*/
std::vector<std::pair<int, int>> findParenthesisPairs(const std::string &text, const std::string &parenthesisType) {
    char openChar, closeChar;
    if (parenthesisType == "()") {
        openChar = '(';
        closeChar = ')';
    } else if (parenthesisType == "[]") {
        openChar = '[';
        closeChar = ']';
    } else if (parenthesisType == "{}") {
        openChar = '{';
        closeChar = '}';
    } else if (parenthesisType == "<>") {
        openChar = '<';
        closeChar = '>';
    } else {
        throw std::runtime_error("Invalid parenthesis type provided.");
    }
    std::stack<int> stack;
    std::vector<std::pair<int, int>> pairs;

    for (int i = 0; i < text.length(); ++i) {
        if (text[i] == openChar) {
            stack.push(i);
        } else if (text[i] == closeChar) {
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

// Function: splitStringAtFirstSpace
// Purpose: Splits a given string into two parts based on the first occurrence of a space character.
// Parameters:
//    - input: The string to be split.
// Returns: A vector containing two strings if a space is found; otherwise, a vector containing the original string.
/* 
Usage Example:
// To split the string "Hello World" at the first space
std::string myString = "Hello World";
std::vector<std::string> split = splitStringAtFirstSpace(myString);
// 'split' will contain two strings: "Hello" and "World"
*/
std::vector<std::string> splitStringAtFirstSpace(const std::string& input) {
    std::vector<std::string> result;
    size_t spacePos = input.find(" ");

    // Check if a space was found
    if (spacePos != std::string::npos) {
        // Split the string at the space position
        result.push_back(input.substr(0, spacePos)); // First part
        result.push_back(input.substr(spacePos + 1)); // Second part
    } else {
        // No space found, return the original string
        result.push_back(input);
    }

    return result;
}

// Function: concatenateWithCustomSeparator
// Purpose: Concatenates all elements of a string vector into a single string, 
//          inserting a custom separator string between each element.
// Parameters:
//    - strings: A vector of strings to be concatenated.
//    - separator: A string used as a separator between each element.
// Returns: A single string that is the result of concatenating all elements of 
//          the input vector, separated by the custom separator string.
/*
Usage Example:
// To concatenate a vector of strings into one string with a custom separator
std::vector<std::string> vec{"This", "is", "a", "test"};
std::string concatenatedString = concatenateWithCustomSeparator(vec, ", ");
// 'concatenatedString' will be "This, is, a, test"
*/
std::string concatenateWithCustomSeparator(const std::vector<std::string>& strings, const std::string& separator) {
    std::string result;

    for (size_t i = 0; i < strings.size(); ++i) {
        result += strings[i];
        if (i != strings.size() - 1) { // Check to avoid adding the separator after the last element
            result += separator;
        }
    }

    return result;
}

// Function: flattenAffineMap
// Purpose: Transforms an input string according to an affine map pattern. It supports handling multiple inputs
//          and optional sections in the affine map pattern. The function parses both the affine map pattern
//          and the input string, replaces variables in the pattern with corresponding values from the input string,
//          and computes the resulting string.
// Parameters:
//    - mapPattern: A string representing the affine map pattern. It is expected to follow the format 
//                  "affine_map<(d0, d1, ...)[s0, s1, ...] -> (outputPattern)>". 
//                  `d` values are required inputs, `s` values are optional.
//    - inputString: A string representing the input to the affine map. It is expected to follow the format 
//                   "#affine_mapX(d0Value, d1Value, ...)[s0Value, s1Value, ...]", where X is an optional map identifier.
// Returns: A string that is the result of applying the affine map pattern to the input string. If the pattern or input 
//          string is invalid, it returns an error message.
/* 
Usage Example:
// To apply an affine map pattern to an input string
std::string mapPattern = "#affine_map42 = affine_map<(d0, d1)[s0, s1] -> (d0 + d1 + s0 + s1)>";
std::string inputString = "#affine_map42(%arg1, %arg2)[%0, %1]";
std::string result = flattenAffineMap(mapPattern, inputString);
// 'result' will be the string "%arg1 + %arg2 + %0 + %1"
*/
std::string flattenAffineMap(const std::string& mapPattern, const std::string& inputString) {
    // Updated regex to handle dynamic map names
    std::regex mapRegex(R"(#(\w+) = affine_map<\((.*?)\)(?:\[(.*?)\])? -> \((.*?)\)>)");
    std::regex inputRegex(R"(#(\w+)\((.*?)\)(?:\[(.*?)\])?)");
    std::smatch mapMatch, inputMatch;

    // Extracting map pattern
    if (!std::regex_search(mapPattern, mapMatch, mapRegex) || mapMatch.size() < 5) {
        return "Invalid map pattern";
    }

    // Check if the map names match
    std::string mapName = mapMatch[1];
    if (!std::regex_search(inputString, inputMatch, inputRegex) || inputMatch.size() < 4 || inputMatch[1] != mapName) {
        return "Invalid input string or map name mismatch";
    }

    std::vector<std::string> dValues = split(mapMatch[2], ",");
    std::vector<std::string> sValues;
    if (mapMatch.size() > 3 && !mapMatch[3].str().empty()) {
        sValues = split(mapMatch[3], ",");
    }
    std::string outputPattern = mapMatch[4];

    std::vector<std::string> dInputValues = split(inputMatch[2], ",");
    std::vector<std::string> sInputValues;
    if (inputMatch.size() > 3 && !inputMatch[3].str().empty()) {
        sInputValues = split(inputMatch[3], ",");
    }

    // Replace and calculate
    std::string result = outputPattern;
    for (size_t i = 0; i < dValues.size(); ++i) {
        if (i < dInputValues.size()) {
            result = std::regex_replace(result, std::regex(dValues[i]), dInputValues[i]);
        }
    }

    for (size_t i = 0; i < sValues.size(); ++i) {
        if (i < sInputValues.size()) {
            result = std::regex_replace(result, std::regex(sValues[i]), sInputValues[i]);
        }
    }

    return concatenateWithCustomSeparator(split(result, " "), " ");
}


// Function: canBeParsedAsType
// Purpose: Checks if a string can be successfully parsed as a specified numeric type without throwing exceptions.
//          The function attempts to parse the string and checks if the entire string has been consumed in this process.
// Parameters:
//    - str: The string to check for numeric convertibility.
//    - type: A string specifying the desired numeric type ("i32" for int32_t, "i64" for int64_t, 
//            "f32" for float, "f64" for double).
// Returns: 'true' if the string can be parsed as the specified type with no leftover characters; 'false' otherwise.
// Usage example:
// std::string example = "1234";
// bool isInt32 = canBeParsedAsType(example, "i32");
// if (isInt32) {
//     std::cout << example << " can be parsed as int32_t." << std::endl;
// } else {
//     std::cout << example << " cannot be parsed as int32_t." << std::endl;
// }
bool canBeParsedAsType(const std::string& str, const std::string& type) {
    assert(type == "i32" || type == "i64" || type == "f32" || type == "f64");

    try {
        size_t pos;

        if (type == "i32") {
            std::stoi(str, &pos);
            if (pos == str.length()) return true;
        } else if (type == "i64") {
            std::stoll(str, &pos);
            if (pos == str.length()) return true;
        } else if (type == "f32") {
            std::stof(str, &pos);
            if (pos == str.length()) return true;
        } else if (type == "f64") {
            std::stod(str, &pos);
            if (pos == str.length()) return true;
        }
    }
    catch (const std::invalid_argument& e) {
        // Not a valid numeric string for the specified type
    }
    catch (const std::out_of_range& e) {
        // The number is out of range for the specified type
    }

    return false;
}


/* 
Function: have_no_intersection
Purpose: Determines if a set of ranges have no intersections among them. 
         It checks every pair of ranges to see if they overlap.

Parameters:
   - ranges: A vector of pairs, where each pair consists of two values (of type T) 
             representing a range. The pair's first value is the start of the range, 
             and the second value is the end of the range.

Returns: A boolean value. Returns true if no intersections are found among the ranges, 
         false otherwise.

Description: The function iterates through each pair of ranges. For each pair, it checks 
             if there is an overlap. An overlap occurs if the end of the first range is 
             greater than or equal to the start of the second range and the start of the 
             first range is less than or equal to the end of the second range. If any 
             intersection is found, the function immediately returns false. If no 
             intersections are found after checking all pairs, the function returns true.

Example:
    vector<pair<int, int>> ranges = {{1, 3}, {4, 5}, {6, 8}};
    bool result = have_no_intersection(ranges);
    // 'result' will be true since there are no overlapping ranges.
*/
template <typename T>
bool have_no_intersection(const std::vector<std::pair<T, T>>& ranges) {
    for (size_t i = 0; i < ranges.size(); ++i) {
        for (size_t j = i + 1; j < ranges.size(); ++j) {
            // Unpack the ranges
            T a = ranges[i].first;
            T b = ranges[i].second;
            T c = ranges[j].first;
            T d = ranges[j].second;

            // Check for intersection
            if (a <= d && b >= c) {
                return false; // Intersection found
            }
        }
    }
    return true; // No intersections found
}

/* 
Function: find_brace_pairs
Purpose: Identifies and pairs opening and closing braces in a given string of code. 
         The function goes through each line of the code, tracking the positions of 
         opening and closing braces.

Parameters:
   - code: A string representing the code to be analyzed. The string is expected to 
           contain multiple lines of code, possibly with braces '{' and '}'.

Returns: A vector of pairs, where each pair consists of two pairs of integers. The first 
         pair in each element denotes the line number and character number of an opening 
         brace, and the second pair denotes the line number and character number of the 
         corresponding closing brace.

Description: The function uses a stack to track opening braces. When an opening brace 
             is encountered, its position is pushed onto the stack. When a closing brace 
             is encountered, the function checks the stack. If the stack is not empty, 
             it pops the top element (the last unmatched opening brace) and forms a pair 
             with the closing brace. If the stack is empty (no matching opening brace), 
             it reports an unmatched closing brace. After parsing, the function also checks 
             for any unmatched opening braces remaining in the stack.

Example:
    string code = "int main() { return 0; }";
    auto brace_pairs = find_brace_pairs(code);
    // 'brace_pairs' will contain pairs indicating the positions of the matched braces.
*/
std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> find_brace_pairs(const std::string& code) {
    std::istringstream iss(code);
    std::string line;
    std::stack<std::pair<int, int>> stack;
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> pairs;

    int line_num = 0;
    while (std::getline(iss, line)) {
        for (int char_num = 0; char_num < line.length(); ++char_num) {
            char ch = line[char_num];

            // Check for opening brace
            if (ch == '{') {
                stack.push({line_num, char_num});
            }

            // Check for closing brace
            if (ch == '}') {
                if (!stack.empty()) {
                    std::pair<int, int> opening_brace = stack.top();
                    stack.pop();
                    pairs.push_back({opening_brace, {line_num, char_num}});
                } else {
                    std::cout << "Unmatched closing brace at line " << line_num + 1 << std::endl;
                }
            }
        }
        line_num++;
    }

    // Check for unmatched opening braces
    while (!stack.empty()) {
        std::pair<int, int> opening_brace = stack.top();
        stack.pop();
        std::cout << "Unmatched opening brace at line " << opening_brace.first + 1 << std::endl;
    }

    return pairs;
}

/* 
Function: find_match
Purpose: Finds the closest enclosing brace pair for a given line and character position from a set of brace pairs.

Parameters:
   - brace_pairs: A vector of pairs, where each pair consists of two pairs of integers. Each pair represents the line 
                  and character numbers for the opening and closing braces.
   - line_num: A pair of integers representing the line number and character position for which the closest brace pair 
               is to be found.
   - size: An integer representing the total number of characters per line, used for normalization.

Returns: A pair of pairs of integers, representing the closest brace pair that encloses the specified line and character 
         position.

Description: The function first converts each pair of brace positions into a normalized floating-point representation. 
             This normalization is done by adding the line number to the character position divided by the total size. 
             The function then iterates through these normalized pairs to find the pair that most closely encloses the 
             given line number, represented as a floating point. The closest pair is determined based on the brace pair 
             that has the smallest range but still contains the line number. The function asserts that the best match is 
             always found, and returns the corresponding brace pair from the original list.

Note: The function uses assertions to ensure the integrity of the logic, such as ensuring that the ending position of 
      each brace is greater than or equal to its starting position, and that a valid match is found.

Example Usage:
    // Assuming brace_pairs is populated with brace pairs
    std::pair<int, int> line_num = {3, 15}; // Line number and character position
    int size = 100; // Number of characters per line
    auto match = find_match(brace_pairs, line_num, size);
    // 'match' contains the closest enclosing brace pair for the given line number.
*/
std::pair<std::pair<int, int>, std::pair<int, int>> find_match(const std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>>& brace_pairs, std::pair<int, int> line_num, int size) {
    std::pair<float, float> best_match = {0, static_cast<float>(size) + 1};
    int best_match_index = -1;
    std::vector<std::pair<float, float>> digit_pair;

    for (const auto& pair : brace_pairs) {
        digit_pair.emplace_back(
            static_cast<float>(pair.first.first) + static_cast<float>(pair.first.second) / size,
            static_cast<float>(pair.second.first) + static_cast<float>(pair.second.second) / size
        );
    }

    float line_num_float = static_cast<float>(line_num.first) + static_cast<float>(line_num.second) / size;

    for (size_t pair_num = 0; pair_num < digit_pair.size(); ++pair_num) {
        const auto& i = digit_pair[pair_num];
        assert(i.second >= i.first && best_match.second >= best_match.first);
        assert((i.first >= best_match.second) || (i.second <= best_match.first) || (i.first <= best_match.first && i.second >= best_match.second) || (i.first >= best_match.first && i.second <= best_match.second));

        if (line_num_float >= i.first && line_num_float <= i.second && i.first > best_match.first && i.second < best_match.second) {
            best_match = i;
            best_match_index = static_cast<int>(pair_num);
        }
    }

    assert(best_match_index != -1);
    return brace_pairs[static_cast<size_t>(best_match_index)];
}


/* 
Function: replace_whole_word
Purpose: Replaces occurrences of a specific whole word in a string with another word, 
         ensuring that only complete words are replaced.

Parameters:
   - s: The original string where the replacements are to be made.
   - old_word: The word to be replaced.
   - new_word: The word to replace with.

Returns: A modified string where each occurrence of the whole 'old_word' is replaced with 'new_word'.

Description: The function first finds all occurrences of 'old_word' in 's'. For each occurrence, 
             it checks if the word is a whole word by ensuring that the character following 'old_word' 
             is not alphanumeric (i.e., it is either a non-alphanumeric character or the end of the string). 
             If it is a whole word, the function then replaces it with 'new_word'. To avoid issues with 
             changing string length during replacements, it first stores the indices of all occurrences and 
             then performs the replacements in reverse order, starting from the last occurrence.

Note: The function uses a vector to keep track of the indices where replacements need to occur. This is 
      necessary because modifying the string directly in a forward loop could alter the positions of later 
      occurrences.

Example Usage:
    std::string original = "This is a test. Testing is fun.";
    std::string result = replace_whole_word(original, "test", "experiment");
    // 'result' will be "This is a experiment. Testing is fun." - 'test' is replaced, but 'Testing' is not.
*/
std::string replace_whole_word(const std::string& s, const std::string& old_word, const std::string& new_word) {
    std::string result = s;
    size_t start = 0;
    std::vector<size_t> indices;

    // Find all occurrences of old_word
    while (true) {
        start = result.find(old_word, start);
        if (start == std::string::npos) {
            break;
        }
        indices.push_back(start);
        start += 1;
    }

    // Replace each occurrence of the whole word
    for (auto i = indices.rbegin(); i != indices.rend(); ++i) {
        if (*i + old_word.length() >= result.length() || (!std::isalnum(result[*i + old_word.length()]) && result[*i + old_word.length()]!='_' )) {
            result = result.substr(0, *i) + new_word + result.substr(*i + old_word.length());
        }
    }

    return result;
}


/* 
Function: countSubstrOccurences
Purpose: Counts the number of occurrences of a substring within a given string.

Parameters:
   - str: The string in which to search for occurrences of the substring.
   - substr: The substring whose occurrences are to be counted in 'str'.

Returns: The number of times 'substr' occurs within 'str'.

Description:
   - The function initializes a count variable to 0, which will hold the number of occurrences.
   - It searches for the first occurrence of 'substr' in 'str' using the find method.
   - While the position of 'substr' in 'str' is not npos (indicating the end of the string), 
     the function increments the count, and searches for the next occurrence of 'substr' starting 
     from the position just after the previous occurrence.
   - This process continues until no more occurrences are found, at which point the function 
     returns the total count.

Example Usage:
    std::string mainStr = "hello, world, hello universe";
    std::string toFind = "hello";
    size_t numOccurrences = countSubstrOccurences(mainStr, toFind);
    std::cout << "Number of occurrences: " << numOccurrences << std::endl;
    // This would output "Number of occurrences: 2" as 'hello' occurs twice in 'mainStr'.

Note:
   - The function uses std::string::find to locate the occurrences of 'substr' in 'str', 
     which performs a case-sensitive search.
   - If 'substr' is an empty string, the function returns 0, as no meaningful occurrences 
     can be counted.
*/
size_t countSubstrOccurences(const std::string& str, const std::string& substr) {
    size_t count = 0;
    size_t pos = str.find(substr);
    while (pos != std::string::npos) {
        count++;
        pos = str.find(substr, pos + substr.size());
    }
    return count;
}

/* 
Function: apply_replacements
Purpose: Applies a series of replacements to a given vector of strings (typically lines of code or text).

Parameters:
   - lines: A reference to a vector of strings, each string representing a line where replacements need to be made.
   - replacements: A map where each key is a string to be replaced, and the value is another map. This inner map
     pairs an integer (line number) with a pair. This pair consists of the new string (replacement) and another pair
     indicating the range of lines over which this replacement should occur.

Returns: The modified vector of strings after applying all replacements.

Description:
   - The function iterates over each replacement specified in the 'replacements' map.
   - For each replacement, it further iterates over the associated inner map, which contains specific line numbers
     and replacement details.
   - The replacement is applied to each specified line within the range indicated by the inner pair of integers.
   - The function uses 'replace_whole_word' to perform replacements, ensuring only whole words are replaced.
   - After applying all replacements, the modified 'lines' vector is returned.

Example Usage:
    std::vector<std::string> lines = { ... }; // Vector containing lines of text/code
    std::map<std::string, std::map<int, std::pair<std::string, std::pair<std::pair<int, int>, std::pair<int, int>>>>> replacements = { ... };
    std::vector<std::string> modifiedLines = apply_replacements(lines, replacements);
    // Process 'modifiedLines' as needed

Note:
   - The function assumes the existence of a 'replace_whole_word' function which replaces a whole word in a string.
   - The replacements are applied in-place, modifying the original 'lines' vector.
   - This function can handle complex replacement scenarios, such as renaming variables in code while respecting 
     their scope and avoiding partial word replacements.
*/
std::vector<std::string> apply_replacements(
    std::vector<std::string>& lines, 
    const std::map<std::string, std::map<int, std::pair<std::string, std::pair<std::pair<int, int>, std::pair<int, int>>>>>& replacements, size_t length) {

    for (const auto& rep : replacements) {
        std::vector<std::pair<float, float>> ranges;
        for (const auto& replacement : replacements.at(rep.first)) {
            auto start = static_cast<float>(replacement.second.second.first.first) + 
                        static_cast<float>(replacement.second.second.first.second) / length;
            auto end = static_cast<float>(replacement.second.second.second.first) + 
                    static_cast<float>(replacement.second.second.second.second) / length;
            ranges.push_back(std::make_pair(start, end));
        }
        assert(have_no_intersection(ranges));
        for (const auto& line_num_pair : rep.second) {
            const auto& line_num = line_num_pair.first;
            const auto& new_var = line_num_pair.second.first;
            const auto& matchs = line_num_pair.second.second;
            
            // lines[matchs.first.first] = replace_whole_word(lines[matchs.first.first], rep.first, new_var);
            // lines[matchs.second.first] = replace_whole_word(lines[matchs.second.first], rep.first, new_var);

            lines[matchs.first.first] = lines[matchs.first.first].substr(0, matchs.first.second + 1) + 
                                replace_whole_word(lines[matchs.first.first].substr(matchs.first.second + 1), rep.first, new_var);

            // Perform replacement at the end of the range
            lines[matchs.second.first] = replace_whole_word(lines[matchs.second.first].substr(0, matchs.second.second + 1), rep.first, new_var) + 
                                            lines[matchs.second.first].substr(matchs.second.second + 1);
            for (int i = matchs.first.first + 1; i < matchs.second.first; ++i) {
                lines[i] = replace_whole_word(lines[i], rep.first, new_var);
            }
        }
    }

    return lines;
}


/* 
Function: rename_variables
Purpose: Renames variables in a given MLIR code string. The function operates in two main passes: 
         identifying variables to be renamed, and then replacing their occurrences throughout the code.

Parameters:
   - mlir_code: A string representing the MLIR code in which the variables are to be renamed.

Returns: A string of the MLIR code with variables renamed.

Description:
   - The function first splits the MLir code into lines and iterates through each line to identify 
     variables that need to be renamed. It checks for lines containing an '=' sign and a '%' sign, 
     and not containing 'for' or 'func.func', indicating a variable assignment.
   - Each identified variable is renamed with a new name formed by appending its count to a prefix '%_'.
   - The function then performs a second pass over the code. In this pass, it replaces all occurrences 
     of the original variable names with the new names within their respective scopes. 
   - The scope of a variable is determined using the `find_brace_pairs` and `find_match` functions, 
     which use brace pairing to ascertain the scope of the variable.
   - Finally, the function replaces '%_' with '%' to conform to the MLIR variable naming convention.

Example Usage:
    std::string mlir_code = "module attributes {...}"; // Truncated for brevity
    std::string transformed_code = rename_variables(mlir_code);
    std::cout << transformed_code << std::endl;
    // The output will be the MLIR code with variables renamed according to their scope.

Note:
    The renaming process is sensitive to the structure of the MLIR code and assumes specific formatting 
    and syntax. The function is designed specifically for MLIR code and may not be suitable for other 
    programming languages or code structures.
*/
std::string rename_variables(const std::string& mlir_code) {
    std::istringstream iss(mlir_code);
    std::string line;
    std::vector<std::string> lines;
    int variable_counts = 0;
    int constant_counts = 0;
    int arg_count = 0;
    std::map<std::string, std::map<int, std::pair<std::string, std::pair<std::pair<int, int>, std::pair<int, int>>>>> replacements;

    // Split code into lines
    while (std::getline(iss, line)) {
        lines.push_back(line);
    }
    auto pair = find_brace_pairs(mlir_code);
    std::vector<std::pair<double, double>> digit_pair;
    for (auto& p : pair) {
        digit_pair.emplace_back(std::make_pair(p.first.first + static_cast<double>(p.first.second) / mlir_code.length(),
                                            p.second.first + static_cast<double>(p.second.second) / mlir_code.length()));
    }
    // First pass: Identify variables to be renamed
    for (int line_num = 0; line_num < lines.size(); ++line_num) {
        line = lines[line_num];
        if (line.find("=") != std::string::npos && line.find("%") != std::string::npos) {
            if (line.find("for") == std::string::npos && line.find("func.func") == std::string::npos && line.find("constant") == std::string::npos) {
                // Handle variable renaming
                std::string var_name = line.substr(0, line.find('='));
                var_name.erase(std::remove_if(var_name.begin(), var_name.end(), ::isspace), var_name.end());
                
                if (var_name != "%" + std::to_string(variable_counts)) {
                    std::string new_var_name = "%_" + std::to_string(variable_counts);
                    auto matchs = find_match(find_brace_pairs(mlir_code), std::make_pair(line_num, line.find("%")), mlir_code.length());
                    replacements[var_name][line_num] = std::make_pair(new_var_name, matchs);
                    variable_counts++;
                } else {
                    variable_counts++;
                }
            } else if (line.find("constant") != std::string::npos) {
                // Handle constant renaming
                std::string var_name = line.substr(0, line.find('='));
                var_name.erase(std::remove_if(var_name.begin(), var_name.end(), ::isspace), var_name.end());
                
                if (var_name != "%cst_" + std::to_string(constant_counts)) {
                    std::string new_var_name = "%cst__" + std::to_string(constant_counts);
                    auto matchs = find_match(find_brace_pairs(mlir_code), std::make_pair(line_num, line.find("%")), mlir_code.length());
                    replacements[var_name][line_num] = std::make_pair(new_var_name, matchs);
                    constant_counts++;
                } else {
                    constant_counts++;
                }
            } else if (line.find("func.func") != std::string::npos) {
                // Handle argument count in function definition
                // arg_count += std::count(line.begin(), line.end(), 'arg');
                arg_count += countSubstrOccurences(line, "arg");
            } else if (line.find("for") != std::string::npos) {
                std::string var_name = split(line," ").at(1);
                var_name.erase(std::remove_if(var_name.begin(), var_name.end(), ::isspace), var_name.end());
                if (var_name != "%arg" + std::to_string(arg_count)) {
                    std::string new_var_name = "%arg_" + std::to_string(arg_count);
                    int index = -1;
                    double smallest = std::numeric_limits<double>::infinity();
                    for (size_t i = 0; i < digit_pair.size(); ++i) {
                        if (digit_pair[i].first > line_num && digit_pair[i].first < smallest) {
                            index = i;
                            smallest = digit_pair[i].first;
                        }
                    }
                    if (index != -1) {
                        auto matchs = pair[index];
                        replacements[var_name][line_num] = std::make_pair(new_var_name, matchs);
                        lines[line_num] = replace_whole_word(lines[line_num], var_name, new_var_name);
                    }
                    arg_count++;
                }
                else {
                    arg_count++;
                }
            }
        }
    }


    // Second pass: Replace all occurrences of the variables
    lines = apply_replacements(lines, replacements,mlir_code.length());
    // for (auto& rep : replacements) {
    //     for (auto& line_num_pair : rep.second) {
    //         auto& line_num = line_num_pair.first;
    //         auto& new_var = line_num_pair.second.first;
    //         auto& matchs = line_num_pair.second.second;
            
    //         lines[matchs.first.first] = replace_whole_word(lines[matchs.first.first], rep.first, new_var);
    //         lines[matchs.second.first] = replace_whole_word(lines[matchs.second.first], rep.first, new_var);
    //         for (int i = matchs.first.first + 1; i < matchs.second.first; ++i) {
    //             lines[i] = replace_whole_word(lines[i], rep.first, new_var);
    //         }
    //     }
    // }
    

    // Final step: Replace "%_" with "%", "%cst__" with "%cst_", "%arg_" with "%arg"
    for (auto& line : lines) {
        size_t pos = 0;
        while ((pos = line.find("%_", pos)) != std::string::npos) {
            line.replace(pos, 2, "%");
            pos += 1;
        }
    }

    for (auto& line : lines) {
        size_t pos = 0;
        while ((pos = line.find("%cst__", pos)) != std::string::npos) {
            line.replace(pos, 6, "%cst_");
            pos += 1;
        }
    }

    for (auto& line : lines) {
        size_t pos = 0;
        while ((pos = line.find("%arg_", pos)) != std::string::npos) {
            line.replace(pos, 5, "%arg");
            pos += 1;
        }
    }
    // std::cout<<"------------------------"<<std::endl;
    // utilities::print(lines);
    // std::cout<<"------------------------"<<std::endl;
    
    // Reassemble the lines into a single string
    std::ostringstream oss;
    for (const auto& line : lines) {
        oss << line << "\n";
    }

    return oss.str();
}

std::string apply_placeholder(const std::string& mlir_code) {
    std::istringstream iss(mlir_code);
    std::string line;
    std::vector<std::string> lines;
    std::map<std::string, std::map<int, std::pair<std::string, std::pair<std::pair<int, int>, std::pair<int, int>>>>> replacements;

    // Split code into lines
    while (std::getline(iss, line)) {
        lines.push_back(line);
    }
    auto pair = find_brace_pairs(mlir_code);
    std::vector<std::pair<double, double>> digit_pair;
    for (auto& p : pair) {
        digit_pair.emplace_back(std::make_pair(p.first.first + static_cast<double>(p.first.second) / mlir_code.length(),
                                            p.second.first + static_cast<double>(p.second.second) / mlir_code.length()));
    }

    for (int line_num = 0; line_num < lines.size(); ++line_num) {
        line = lines[line_num];
        if (line.find("for") != std::string::npos) {
            std::string var_name = split(line," ").at(1);
            var_name.erase(std::remove_if(var_name.begin(), var_name.end(), ::isspace), var_name.end());
            std::string new_var_name = "Placeholder4" + var_name;
            int index = -1;
            double smallest = std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < digit_pair.size(); ++i) {
                if (digit_pair[i].first > line_num && digit_pair[i].first < smallest) {
                    index = i;
                    smallest = digit_pair[i].first;
                }
            }
            if (index != -1) {
                auto matchs = pair[index];
                replacements[var_name][line_num] = std::make_pair(new_var_name, matchs);
            }
        }
    }

    // for (int line_num = 0; line_num < lines.size(); ++line_num) {
    //     std::cout<<line_num<<" "<<lines[line_num]<<std::endl;
    // }

    // print(replacements);
    lines = apply_replacements(lines, replacements, mlir_code.length());

    std::ostringstream oss;
    for (const auto& line : lines) {
        oss << line << "\n";
    }

    return oss.str();
}

/* 
Function: prefixToInfix
Purpose: Converts a mathematical expression from prefix notation to infix notation.

Parameters:
   - prefix: A string representing the mathematical expression in prefix notation.

Returns: A string representing the equivalent expression in infix notation.

Description:
   - The function iterates over the given prefix expression in reverse.
   - When an operator (one of '*', '/', '+', or '-') is encountered, the function pops two elements from the stack, 
     which are operands, forms a new expression by placing the operator between these two operands with appropriate 
     parentheses, and then pushes this new expression back onto the stack.
   - When an operand is encountered (either a number, an alphanumeric variable, or a modulus operator '%'), 
     the function reads the entire operand (which might be more than one character long) and pushes it onto the stack.
   - After the iteration, the final element left in the stack is the infix expression.

Example Usage:
    std::string prefixExp = "(* (+ A B) (- C D))";
    std::string infixExp = prefixToInfix(prefixExp);
    std::cout << infixExp << std::endl; 
    // The output will be "((A + B) * (C - D))".

Note:
    The function assumes that the input prefix expression is valid. It does not handle invalid expressions or operators 
    other than '+', '-', '*', and '/'. It also assumes that there are no spaces in the input prefix expression.
*/
std::string prefixToInfix(const std::string& prefix) {
    std::stack<std::string> stack;

    // Iterate over the expression in reverse
    for (int i = prefix.length() - 1; i >= 0; --i) {
        if (prefix[i] == '*' || prefix[i] == '/' || prefix[i] == '+' || prefix[i] == '-') {
            // Operator, pop two elements from stack
            std::string op1 = stack.top(); stack.pop();
            std::string op2 = stack.top(); stack.pop();

            // Form a new expression and push back to stack
            std::string exp = "(" + op1 + " " + prefix[i] + " " + op2 + ")";
            stack.push(exp);
        } else if (std::isalnum(prefix[i]) || prefix[i] == '%') {
            // Operand, push to stack
            int j = i;
            while (i >= 0 && (std::isalnum(prefix[i]) || prefix[i] == '%')) {
                --i;
            }
            stack.push(prefix.substr(i + 1, j - i));
        }
    }

    // The final element in the stack is the infix expression
    return stack.empty() ? "" : stack.top();
}


/* 
Function: findCorrespondingMap
Purpose: Identifies a mapping from a given unordered_map that corresponds to a specific expression,
         extracts relevant parts of the expression based on this mapping, and returns these parts
         along with the key of the matching mapping.

Parameters:
   - maps: An unordered_map where keys are strings representing the mapping names, and values 
           are strings representing the mappings themselves.
   - expression: A string representing the expression to match against the mappings.

Returns: A tuple containing three elements:
            1. The key of the matching map as a string.
            2. A vector of strings representing the first part of the extracted expression arguments.
            3. A vector of strings representing the second part of the extracted expression arguments.

Description:
   - The function starts by splitting the expression into tokens, excluding operators and non-floating point tokens.
   - It then iterates over each pair in the provided maps, splitting and processing the map values to extract 
     tokens, which are then compared with tokens from the expression.
   - For each map, it compares the sorted tokens from the map with those from the expression. If they match 
     in size, it further checks for structural matches by permuting the expression tokens and comparing the 
     restructured expression against the map expression.
   - When a match is found, it assembles a tuple with the map key and the corresponding parts of the expression 
     that match the map's structure and adds it to a result set.
   - If exactly one match is found in the result set, it returns this result. Otherwise, it throws an error.

Example Usage:
    std::unordered_map<std::string, std::string> maps = {
            {"#map1", "#map1 = affine_map<()[s0] -> (s0 - 1)>"},
            {"#map2", "#map2 = affine_map<()[s0] -> (s0 + 1)>"},
            {"#map3", "#map4 = affine_map<(s1,s2)[s0] -> ((((s1 + 1) * s0) / s2) - 3)>"},
            {"#map4", "#map3 = affine_map<()[s1,s0] -> ((((s1 + 1) * s0) / s1) - 3)>"}
        };

        std::string expression = "((((%2 + 1) * %1) / %3) - 3)";
        try {
            auto matchingMap = findCorrespondingMap(maps, expression);
            std::cout << "Matching map: " << std::get<0>(matchingMap) << std::endl;
            std::cout << "Arguments (): ";
            for (const auto &arg : std::get<1>(matchingMap)) {
                std::cout << arg << " ";
            }
            std::cout << std::endl;
            std::cout << "Arguments []: ";
            for (const auto &arg : std::get<2>(matchingMap)) {
                std::cout << arg << " ";
            }
            std::cout << std::endl;
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
        }

    Output is:
    Matching map: #map3
    Arguments (): %2 %3 
    Arguments []: %1

Note:
    - This function relies on the existence of utility functions like 'split', 'canBeParsedAsType', 
      and 'getContentInFirstPair', which are not defined in this code snippet.
    - The function is specifically tailored for mapping expressions to a particular format of maps, 
      making it highly specialized for cases where such mapping and comparison are needed.
    - The implementation involves complex logic with sorting, permutations, and string manipulation, 
      which may be computationally intensive for large sets of maps or complex expressions.
*/
std::tuple<std::string, std::vector<std::string>, std::vector<std::string>> findCorrespondingMap(const std::unordered_map<std::string, std::string> &maps, const std::string &expression) {
    auto expressionTokens = split(expression, "() ");
    std::vector<std::string> expressionArgs;

    for (const auto &token : expressionTokens) {
        if (token.find_first_of("+-*/") == std::string::npos && !canBeParsedAsType(token,"f64")) {
            expressionArgs.push_back(token);
        }
    }

    std::vector<std::tuple<std::string, std::vector<std::string>, std::vector<std::string>>> resultSet;
    for (const auto &mapPair : maps) {
        auto mapKey = mapPair.first;
        auto mapValue = mapPair.second;

        auto mapExpr = mapValue.substr(mapValue.find("->") + 2);
        if (mapExpr.back() == '>') {
            mapExpr.pop_back();
        }

        auto mapTokens = split(mapExpr, "()<> ");
        std::vector<std::string> mapargs1;
        for (const auto &token : mapTokens) {
            if (token.find_first_of("+-*/") == std::string::npos && !canBeParsedAsType(token,"f64")) {
                mapargs1.push_back(token);
            }
        }
        auto mapargs_parentheses=split(getContentInFirstPair(mapValue,0), ", ");
        auto mapargs_SqBrackets=split(getContentInFirstPair(mapValue,1), ", ");
        std::vector<std::string> mapargs0;
        mapargs0.insert(mapargs0.end(), mapargs_parentheses.begin(), mapargs_parentheses.end());
        mapargs0.insert(mapargs0.end(), mapargs_SqBrackets.begin(), mapargs_SqBrackets.end());
        
        // for (auto i : _mapargs0_) {
        //     std::cout << i <<" ";
        // }
        // std::cout<<std::endl;

        std::sort(mapargs1.begin(), mapargs1.end());
        mapargs1.erase(std::unique(mapargs1.begin(), mapargs1.end()), mapargs1.end());
        std::sort(mapargs1.begin(), mapargs1.end());
        std::sort(mapargs0.begin(), mapargs0.end());
        assert(mapargs0==mapargs1);
        std::vector<std::string> mapargs;
        mapargs.insert(mapargs.end(), mapargs_parentheses.begin(), mapargs_parentheses.end());
        mapargs.insert(mapargs.end(), mapargs_SqBrackets.begin(), mapargs_SqBrackets.end());
        std::sort(expressionArgs.begin(), expressionArgs.end());
        expressionArgs.erase(std::unique(expressionArgs.begin(), expressionArgs.end()), expressionArgs.end());
        if (expressionArgs.size() == mapargs.size()) {
            do {
                std::string tempExpr = expression;
                for (size_t i = 0; i < expressionArgs.size(); ++i) {
                    size_t pos;
                    while ((pos = tempExpr.find(expressionArgs[i])) != std::string::npos) {
                        tempExpr.replace(pos, expressionArgs[i].size(), mapargs[i]);
                    }
                }
                tempExpr.erase(std::remove(tempExpr.begin(), tempExpr.end(), ' '), tempExpr.end());
                mapExpr.erase(std::remove(mapExpr.begin(), mapExpr.end(), ' '), mapExpr.end());
// str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
                if (tempExpr == mapExpr) {
                    std::vector<std::string> firstPart(expressionArgs.begin(), expressionArgs.begin() + mapargs_parentheses.size());
                    std::vector<std::string> secondPart(expressionArgs.begin() + mapargs_parentheses.size(), expressionArgs.end());
                    resultSet.push_back(std::make_tuple(mapKey, firstPart, secondPart));
                }
            } while (std::next_permutation(expressionArgs.begin(), expressionArgs.end()));
        }
    }

    if (resultSet.size() == 1) {
        return resultSet.at(0);
    } else {
        throw std::runtime_error("No matching map found or multiple matches found");
    }
}

/* 
Function: checkPair
Purpose: Checks for specific types of paired characters (like parentheses) in a string 
         and validates their placement based on certain conditions.

Parameters:
   - text: The string in which the paired characters are to be checked.
   - type: An integer representing the type of paired characters to check. 
           The types are: 0 for '()', 1 for '{}', 2 for '[]', and 3 for '<>'.

Returns: A boolean value. Returns true if the conditions for the paired characters 
         are met, false otherwise.

Description:
   - The function first sets the open and close characters based on the 'type' provided.
   - It then calls a helper function `findParenthesisPairs`, which is assumed to return 
     a vector of pairs of integers. Each pair in the vector represents the positions of 
     the opening and closing characters in the 'text'.
   - The function checks if the outermost pair encloses the entire text (i.e., the first 
     character is the opening character and the last character is the closing character).
   - If there is only one such pair, it returns true.
   - If there are more pairs, it further checks the placement of these pairs relative to 
     whitespace characters. It returns true if any pair does not have whitespace characters 
     immediately outside of it.

Example Usage:
    std::string text = "(This is a test)";
    bool isValid = checkPair(text, 0);
    std::cout << (isValid ? "Valid" : "Invalid") << std::endl;
    // This would output "Valid" if 'text' meets the conditions for type 0 (parentheses).

Note:
    - The function assumes the existence of a helper function `findParenthesisPairs` which is 
      not provided in the snippet.
    - The function's logic and return value are based on specific conditions regarding the placement 
      and number of pairs. These conditions may need to be adjusted depending on the intended use case.
    - The function returns false for any 'type' value not in the range 0-3.
*/
bool checkPair(const std::string& text, int type) {
    char openChar, closeChar; std::string brackets;

    // Set the type of parentheses to check
    switch (type) {
        case 0: openChar = '('; closeChar = ')';brackets="()"; break;
        case 1: openChar = '{'; closeChar = '}';brackets="{}"; break;
        case 2: openChar = '['; closeChar = ']';brackets="[]"; break;
        case 3: openChar = '<'; closeChar = '>';brackets="<>"; break;
        default: return false; // Invalid type
    }
    
    auto parenthesisPairs = findParenthesisPairs(text,brackets);
    if (parenthesisPairs.size()>0){
        if (parenthesisPairs.back().first==0 && parenthesisPairs.back().second==text.size()-1){
            if (parenthesisPairs.size()==1){
                return true;
            }
            else{
                parenthesisPairs.pop_back();
                for (auto i:parenthesisPairs){
                    if (text.at(i.first-1)!=' ' or text.at(i.second+1)!=' '){
                        return true;
                    }
                }
            }
        }
    }

    return false;
}


/*
int main() {
    std::string txt = "I like bananas";
    txt = replaceString(txt, "bananas", "apples");

    std::cout << txt << std::endl; // Output: I like apples

    return 0;
}
*/
std::string replaceString(std::string subject, const std::string& search, const std::string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
    return subject;
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