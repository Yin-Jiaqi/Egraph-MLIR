#ifndef GRAPH_HPP
#define GRAPH_HPP


#include <boost/graph/adjacency_list.hpp>
#include <boost/variant.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <cassert>
#include <memory>


// Forward declaration of Edge to handle mutual dependency with Vertex
class Edge;
class Vertex;
using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, Vertex*, Edge*>;



// Vertex represents operation in graph
class Vertex {
private:
    // Name of the vertex
    std::string _vname = "";
    // Input edges of the vertex, potentially holding multiple data elements
    std::vector<Edge*> _in_data;
    // Output edges of the vertex, holding a single data element
    std::vector<Edge*> _out_data;
    Graph::vertex_descriptor _descriptor;
    // Corresponding line number in the code or script
    int _line_num;
    // The nesting level of this block in the code
    int _block_level;
    // Dimensionality of the output data
    int _dimension;
    // Mapping to track the type of operation performed by the vertex.
    // Note: Only one operation type can be active per category in this map
    std::unordered_map<std::string, std::unordered_map<std::string, int>> _op_type {
        {"Arith", {{"None",0}, {"Addi",0}, {"Muli",0}, {"Addf",0}, {"Mulf",0}, {"Constant",0}, {"IndexCast",0}}},
        {"Memref", {{"None",0}, {"Load",0}, {"Store",0}}},
        {"Func", {{"None",0}, {"Func",0}, {"Return",0}}},
        {"Scf", {{"None",0}, {"For",0}, {"If",0}}},
        {"Affine", {{"None",0}, {"Forvalue",0}, {"Forcontrol",0}, {"If",0}, {"Load",0}, {"Store",0}}},
        // Source/Sink operations
        {"SS", {{"None",0}, {"Source",0}, {"Sink",0}}},
        // Block operations
        {"Block", {{"Block",0}}}
    };
    // Mapping to define the data type of the vertex.
    // Note: Only one data type can be set at a time in this map
    std::unordered_map<std::string, int> _dtype {
        {{"None",0}, {"Index",0}, {"I32",0}, {"I64",0}, {"F32",0}, {"F64",0}, {"I32Index",0}, {"I64Index",0} }
    };

public:
    ~Vertex() {}
    Vertex() {}
    Vertex(std::string str, std::string dialect, std::string operation, std::string dtype , const std::vector<Edge*>& in, const std::vector<Edge*>& out, int dimension, int line_num, int block_level) 
    : _vname(str),_in_data(in), _out_data(out), _dimension(dimension), _line_num(line_num), _block_level(block_level) {
        checkDialectAndOperation(dialect,operation);
        _op_type[dialect][operation]=1;
        checkdtype(dtype);
        _dtype[dtype]=1;
    }
    
    //Deprecated constructor for Vertex. If the input or output data is not found, initialize with an empty vector of Edge*.
    // Vertex(std::string str) : _vname(str),_in_data(), _out_data() {}
    // Vertex(std::string str, std::string dialect, std::string operation, std::string dtype, int dimension, int line_num, int block_level) 
    //     : _vname(str), _dimension(dimension), _line_num(line_num), _block_level(block_level) {
    //         checkDialectAndOperation(dialect,operation);
    //         _op_type[dialect][operation]=1;
    //         checkdtype(dtype);
    //         _dtype[dtype]=1;
    //         }


    // Validates if the given data type is within the allowed _dtype list.
    void checkdtype(const std::string& dtype) {
        if (_dtype.find(dtype) == _dtype.end()) {
            throw std::runtime_error(std::string("Vertex Dtype must select from: ") + utilities::vectorToString(utilities::mapKeys(_dtype)));
        }
    }

    // Validates if the given dialect and operation are within the allowed _op_type list.
    void checkDialectAndOperation(const std::string& dialect, const std::string& operation) {
        if (_op_type.find(dialect) == _op_type.end()) {
            throw std::runtime_error(std::string("Dialect must select from: ") + utilities::vectorToString(utilities::mapKeys(_op_type)));
        }
        if (_op_type.at(dialect).find(operation) == _op_type.at(dialect).end()) {
            throw std::runtime_error(std::string("Operation must select from: ") + utilities::vectorToString(utilities::mapKeys(_op_type.at(dialect))));
        }
    }

    //Getter function
    int getLineNum() const {
        return _line_num;
    }
    int getBlockLevel() const {
        return _block_level;
    }
    int getDimension() const {
        return _dimension;
    }
    std::string getOpName() const {
        return _vname;
    }
    std::vector<Edge*>& getInput() {
        return _in_data;
    }
    std::vector<Edge*> getOutput() const {
        return _out_data;
    }
    std::unordered_map<std::string, std::unordered_map<std::string, int>> getOp() const {
        return _op_type;
    }
    std::unordered_map<std::string, int> getDtype() const {
        return _dtype;
    }
    Graph::vertex_descriptor getDescriptor() const {
        return _descriptor;
    }


    // Setter function
    void setLineNum(int line_num) {
        _line_num=line_num;
    }
    void setBlockLevel(int block_level) {
        _block_level=block_level;
    }
    void setDimension(int dimension) {
        _dimension=dimension;
    }
    void setName(std::string str) {
        _vname = str;
    }
    void setInput(const std::vector<Edge*>& in) {
        _in_data = in;
    }
    void updateInput(Edge* in) {
        _in_data.push_back(in);
    }
    void updateOutput(Edge* out) {
        _out_data.push_back(out);
    }
    void setOutput(const std::vector<Edge*>& out) {
        _out_data = out;
    }
    void setDescriptor(Graph::vertex_descriptor& descriptor){
        _descriptor=descriptor;
    }
    void setDtype(std::string dtype) {
        checkdtype(dtype);
        _dtype[dtype]=1;
    }
    void setOp(std::string dialect, std::string operation) {
        checkDialectAndOperation(dialect,operation);
        _op_type[dialect][operation]=1;
    }
    void set(std::string str, std::string dialect, std::string operation, std::string dtype, const std::vector<Edge*>& in, const std::vector<Edge*>& out, int dimension, int line_num, int block_level)  {
        checkDialectAndOperation(dialect,operation);
        setName(str); setInput(in); setOutput(out); setOp(dialect, operation); setDtype(dtype); setDimension(dimension), setLineNum(line_num), setBlockLevel(block_level);
    }
};

class Edge {
protected:
    // Edge name
    std::string _ename = "";
    // Originating vertex/operation, 
    Vertex* _in_op;
    // Destination vertices, potentially holding multiple data elements
    std::vector<Vertex*> _out_op;
    // Defines the data type of the edge
    std::unordered_map<std::string, int> _dtype {
        {"None", 0}, {"Index", 0}, {"I32", 0}, {"I64", 0}, {"F32", 0}, {"F64", 0}, {"I32Index", 0}, {"I64Index", 0}, {"Block", 0}, {"Pseudo", 0}
    };
    // Dimension of the edge data
    int _dimension;
    // Line number in the source code/script
    int _line_num;

public:

    ~Edge() {}
    Edge() {}
    Edge(std::string name, Vertex* in, const std::vector<Vertex*>& out, std::string dtype, int dimension, int line_num) 
        : _ename(name), _in_op(in), _out_op(out), _dimension(dimension), _line_num(line_num) {
            checkdtype(dtype);
            _dtype[dtype]=1;
        }

    // Validates if the given dialect and operation are within the allowed _op_type list, same with Vertex::checkdtype
    void checkdtype(const std::string& dtype) {
        if (_dtype.find(dtype) == _dtype.end()) {
            throw std::runtime_error(std::string("Edge Dtype must select from: ") + utilities::vectorToString(utilities::mapKeys(_dtype)));
        }
    }


    // Getter functions
    int getLineNum() const {
        return _line_num;
    }
    int getDimension() const {
        return _dimension;
    }
    std::string getEdgeName() const {
        return _ename;
    }
    std::unordered_map<std::string, int> getDtype() const {
        return _dtype;
    }
    Vertex* getInOp() const {
        return _in_op;
    }
    std::vector<Vertex*> getOutOp() const {
        return _out_op;
    }


    // Setter functions
    void setLineNum(int line_num) {
        _line_num=line_num;
    }
    void setEdgeName(const std::string& name) {
        _ename = name;
    }
    void setDimension(int dimension) {
        _dimension=dimension;
    }
    void setInOp(Vertex* in) {
        _in_op = in;
    }
    void setOutOp(std::vector<Vertex*>& out) {
        _out_op = out;
    }    
    void setDtype(std::string dtype) {
        checkdtype(dtype);
        _dtype[dtype]=1;
    }
    //An overall setter function
    void set(std::string name, Vertex* in, std::vector<Vertex*>& out, std::string dtype, int dimension, int line_num) {
        setEdgeName(name); setInOp(in); setOutOp(out); setDtype(dtype);setDimension(dimension), setLineNum(line_num);
    }

    // Updates the _out_op vector by adding a new operation.
    void updateOutOp(Vertex* op, Graph* g) {
        // Before adding a new operation to _out_op, check if the first element is a "Sink" node.
        // If it is, and as the "Sink" node signifies non-usage by other operations, 
        // remove the "Sink" vertex from _out_op and update the "Sink" node's input data.
        if (_out_op[0]->getOpName() == "Sink") {
            assert(_out_op.size() == 1);
            _out_op.erase(_out_op.begin());
            // Iterate through the input edges of the "Sink" node and remove this edge if found.
            for (size_t i = 0; i < (*g)[0]->getInput().size(); ++i) {
                Edge* inEdge = (*g)[0]->getInput()[i];
                // Compare the edge names to find a match.
                if (_ename == inEdge->getEdgeName()) {
                    (*g)[0]->getInput().erase((*g)[0]->getInput().begin() + i);
                    break;
                }
            }
        }
        // Add the operation to the _out_op vector.
        _out_op.push_back(op);
    }
};


class Boost_Graph {
private:
    // Adjacency list representation of the graph, using Boost library
    Graph _g;
    // Vector for storing all edges present in the graph
    std::vector<Edge*> _edge_list;
    // Mapping of vertices in the graph, categorized by operation types.
    // Each operation type maps to a sub-map of operation names and their corresponding vertices.
    std::unordered_map<std::string, std::unordered_map<std::string,std::vector<Vertex*>>> _vertex_list = {
        {"Arith", {{"None",{}}, {"Addi",{}}, {"Muli",{}}, {"Addf",{}}, {"Mulf",{}}, {"Constant",{}}, {"IndexCast",{}}}},
        {"SS", {{"None",{}}, {"Source",{}}, {"Sink",{}}}},
        {"Memref", {{"None",{}}, {"Load",{}}, {"Store",{}}}},
        {"Func", {{"None",{}}, {"Func",{}}, {"Return",{}}}},
        {"Scf", {{"None",{}}, {"For",{}}, {"If",{}}}},
        {"Affine", {{"None",{}}, {"Forvalue",{}}, {"Forcontrol",{}}, {"If",{}}, {"Load",{}}, {"Store",{}}}},
        {"Block", {{"Block",{}}}}
    };
    // Mapping from vertex names to their corresponding Vertex objects.
    std::map<std::string, Vertex*> _name2Vertex;

    // Mapping associating edge names with their corresponding Edge objects.
    // Different data may share same if they appear in different loops.
    // This map differentiates data based on its occurrence in loops. If data appears only once, 
    // both the first and second string keys represent the data's name. However, if the data appears 
    // multiple times, the second string key is a concatenation of the data name and its occurrence index.
    std::map<std::string, std::map<std::string, Edge*>> _name2Edge;



public:
    Boost_Graph(Graph g) 
        : _g(g){}
    
    ~Boost_Graph() {}

    
    //Getter function
    Graph& getGraph() {
        return _g;
    }
    const std::vector<Edge*>& getEdgeList() const {
        return _edge_list;
    }
    const std::unordered_map<std::string, std::unordered_map<std::string,std::vector<Vertex*>>>& getVertexList() const {
        return _vertex_list;
    }
    const std::map<std::string, Vertex*>& getname2Vertex() const {
        return _name2Vertex;
    }
    std::map<std::string, std::map<std::string, Edge*>>& getname2Edge() {
        return _name2Edge;
    }

    // Checks if a vertex with the specified name exists in the graph.
    bool checkVertex(std::string s) const {
        return _name2Vertex.find(s) != _name2Vertex.end();
    }

    // Level 1 Check: Determines if an edge with the specified name exists in _name2Edge,
    // without considering the occurrence count of the edge.
    bool checkEdgeL1(std::string s) const {
        return _name2Edge.find(s) != _name2Edge.end();
    }

    // Level 2 Check: Determines if an edge exists in _name2Edge while also considering
    // the occurrence count of the edge. It splits the edge name to evaluate its existence.
    bool checkEdgeL2(std::string s) const {
        std::vector<std::string> ename_vec = utilities::split(s, "/");
        return checkEdgeL1(ename_vec[0]) && _name2Edge.at(ename_vec[0]).find(s) != _name2Edge.at(ename_vec[0]).end();
    }

    //Print graph information
    void printGraph(std::ostream& os = std::cout) {
        // Print vertices
        os << "Vertices:" << std::endl;
        auto vertexPair = boost::vertices(_g);
        for (auto vIt = vertexPair.first; vIt != vertexPair.second; ++vIt) {
            Vertex* v = _g[*vIt];
            if (v) {
                os << "Name: " << v->getOpName() << std::endl;
                os << "Dtype: " << utilities::findPositionOfOne(v->getDtype()) << std::endl; // Assuming there's only one dtype
                os << "Dimension: " << v->getDimension() << std::endl;
                os << "#Line: " << v->getLineNum() << std::endl;
                os << "#Block: " << v->getBlockLevel() << std::endl;
                os << "Input Edges: ";
                for (Edge* inEdge : v->getInput()) {
                    os << inEdge->getEdgeName() << " ";
                }
                os << std::endl;
                os << "Output Edges: ";
                for (Edge* outEdge : v->getOutput()) {
                    os << outEdge->getEdgeName() << " ";
                }
                os << std::endl;
                os << std::endl;
            }
        }

        // Print edges
        os << "\nEdges:" << std::endl;
        for (const auto& e : _edge_list){
        // Deprecarted version for loop for edge_list, this will print the dulicated edge multiple times
        // auto edgePair = boost::edges(_g);
        // for (auto eIt = edgePair.first; eIt != edgePair.second; ++eIt) {
        //     Edge* e = _g[*eIt];
        //     Vertex* srcVertex = _g[boost::source(*eIt, _g)];
        //     Vertex* tgtVertex = _g[boost::target(*eIt, _g)];

            os << "Edge Name: " << e->getEdgeName() << std::endl;
            os << "Dtype: " << utilities::findPositionOfOne(e->getDtype()) << std::endl; // Assuming there's only one dtype
            os << "Dimension: " << e->getDimension() << std::endl;
            os << "Source Vertex: " << e->getInOp()->getOpName() << std::endl;
            os << "#Line: " << e->getLineNum() << std::endl;
            os << "Target Vertices: ";
            for (Vertex* tgt : e->getOutOp()) {
                os << tgt->getOpName() << " ";
            }
            os << std::endl;
            os << std::endl;
        }
        os << "\n" << std::endl;
    }


    // Add vertex into graph
    void addOrUpdateVertex(Vertex* v) {
        if (!checkVertex(v->getOpName())) {
            Graph::vertex_descriptor desc = boost::add_vertex(v,_g);
            _g[desc] = v;
            v->setDescriptor(desc);
            _name2Vertex[v->getOpName()]=v;
            std::pair<std::string, std::string> position= utilities::findPositionOfOne(v->getOp());
            _vertex_list[position.first][position.second].push_back(v);
        }
    }
    

    // Add edge to graph
    void addEdge2Graph(Edge* e) {
        auto ename=e->getEdgeName();
	    if (!checkEdgeL2(ename)) {
            // Check and possibly add the source and target vertex.
            Vertex* src = e->getInOp();
            std::vector<Vertex*> tgt = e->getOutOp();
            addOrUpdateVertex(src);
            for (auto& v : tgt) {
                addOrUpdateVertex(v);
            }
            // Add the edge to boost graph
            if (!tgt.empty()){
                boost::add_edge(src->getDescriptor(), tgt.at(0)->getDescriptor(), e, _g);
            }
            // Update _edge_list and _name2Edge
            _edge_list.push_back(e);
            _name2Edge[utilities::split(ename,"/").at(0)][e->getEdgeName()]=e;
		}
        else{
            throw std::runtime_error(std::string("Edge exists, try update_edge_to_graph ") + ename);
        }
    }

    // Add output operation for an existed edge
    void updateEdge2Graph(std::string ename, Vertex* op, size_t edge_index) {
        //Level-2 check if the edge name exist
        if (!checkEdgeL2(ename)) {
            throw std::runtime_error(ename + std::string(" does not exist. Try to use add_edge_to_graph function"));
        }
        else{
            // Retrive the edge object
            std::vector<std::string> ename_vec = utilities::split(ename,"/");
            Edge* edge;
            if (ename_vec.size() == 1 || ename_vec.size() == 2)
                edge=_name2Edge[ename_vec[0]][ename];
            else
                throw std::runtime_error(std::string("Incorrect edge name for update: ") + ename);
            addOrUpdateVertex(op);
            edge->updateOutOp(op, &_g);
            boost::add_edge(edge->getInOp()->getDescriptor(), op->getDescriptor(), edge, _g);
            op->getInput()[edge_index]=edge;
        }
    }


};


// For, if, function and all other structure that contain “{}” will be tied to this block
// Record the information for the block
class Block {
private:
    std::string _bname;
    // Stores the starting and ending positions of a block. The 'first' member denotes the starting line number, 
    // while the 'second' member indicates the line number where the block concludes.
    std::unique_ptr<int[]> _pos;
    // _iter is a string array used to store information about loop parameters: start, end, and step.
    // Each element in this array can be an empty string (""), "0", or a variable like "%1".
    std::unique_ptr<std::string[]> _iter;
    int _index;
    Vertex* _knob;
    std::vector<Edge*> _in_edge;

public:
    // Default constructor
    Block() 
        : _bname(""), _index(0), _pos(std::make_unique<int[]>(2)), _iter(std::make_unique<std::string[]>(3)), _knob(nullptr)
    {
        _pos[0] = 0; _pos[1] = 0;
        _iter[0] = ""; _iter[1] = ""; _iter[2] = "";
    }
    // Parameterized constructor
    Block(const std::string& bname, int idx, int x, int y, std::string iter1, std::string iter2, std::string iter3, Vertex* knob = nullptr) 
        : _bname(bname), _index(idx), _pos(std::make_unique<int[]>(2)), _iter(std::make_unique<std::string[]>(3)), _knob(knob)
    {
        _pos[0] = x; _pos[1] = y;
        _iter[0] = iter1; _iter[1] = iter2; _iter[2] = iter3;
    }
    ~Block() {
    }

    void addInEdge(Edge* operation) {
        _in_edge.push_back(operation);
    }

    // Setters
    void setIndex(int idx) {
        _index = idx;
    }

    void setKnob(Vertex* knob) {
        _knob = knob;
    }

    void setBName(const std::string& bname) {
        _bname = bname;
    }

    void setPos(int x, int y) {
        _pos[0] = x;
        _pos[1] = y;
    }

    void setPosend(int y) {
        _pos[1] = y;
    }

    void setIter(std::string iter1, std::string iter2, std::string iter3) {
        _iter[0] = iter1;
        _iter[1] = iter2;
        _iter[2] = iter3;
    }

    void set(const std::string& bname, int idx, int x, int y, std::string iter1, std::string iter2, std::string iter3, Vertex* knob) {
        setBName(bname);
        setIndex(idx);
        setPos(x, y);
        setIter(iter1, iter2, iter3);
        setKnob(knob);
    }

    // Getters

    int getIndex() const {
        return _index;
    }

    Vertex* getKnob() const {
        return _knob;
    }

    std::string getBName() const {
        return _bname;
    }

    int* getPos() const {
        return _pos.get();
    }

    std::string* getIter() const {
        return _iter.get();
    }

    std::vector<Edge*>& getInEdge() {
        return _in_edge;
    }
};

#endif