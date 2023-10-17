#ifndef GRAPH_HPP
#define GRAPH_HPP


#include <boost/graph/adjacency_list.hpp>
#include <boost/variant.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <cassert>

class Edge;
class Vertex;
using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, Vertex*, Edge*>;

// Forward declaration of Edge to handle mutual dependency with Vertex


// Base Vertex class representing a node in the graph
class Vertex {
private:
    std::string _vname = "";
    std::vector<Edge*> _in_data;
    std::vector<Edge*> _out_data;
    Graph::vertex_descriptor _descriptor;
    int _dimension;

    std::unordered_map<std::string, std::unordered_map<std::string, int>> _op_type {
        {"Arith", {{"None",0}, {"Addi",0}, {"Muli",0}, {"Addf",0}, {"Mulf",0}, {"Constant",0}, {"IndexCast",0}}},
        {"SS", {{"None",0}, {"Source",0}, {"Sink",0}}},
        {"Memref", {{"None",0}, {"Load",0}, {"Store",0}}},
        {"Func", {{"None",0}, {"Func",0}, {"Return",0}}},
        {"Scf", {{"None",0}, {"For",0}, {"If",0}}}
        };

    std::unordered_map<std::string, int> _dtype {
        {{"None",0}, {"Index",0}, {"I32",0}, {"I64",0}, {"F32",0}, {"F64",0}, {"I32Index",0}, {"I64Index",0} }
        };
public:
    ~Vertex() {}  // Virtual destructor to ensure proper cleanup of derived objects
    
    Vertex() {}

    Vertex(std::string str) 
    : _vname(str),_in_data(), _out_data() {}

    Vertex(std::string str, std::string dialect, std::string operation, std::string dtype, int dimension) 
        : _vname(str), _dimension(dimension) {
            checkDialectAndOperation(dialect,operation);
            _op_type[dialect][operation]=1;
            checkdtype(dtype);
            _dtype[dtype]=1;
            }

    Vertex(std::string str, std::string dialect, std::string operation, std::string dtype , const std::vector<Edge*>& in, const std::vector<Edge*>& out, int dimension) 
        : _vname(str),_in_data(in), _out_data(out), _dimension(dimension) {
            checkDialectAndOperation(dialect,operation);
            _op_type[dialect][operation]=1;
            checkdtype(dtype);
            _dtype[dtype]=1;
            }

    void checkdtype(const std::string& dtype) {
        if (_dtype.find(dtype) == _dtype.end()) {
            throw std::runtime_error(std::string("Vertex Dtype must select from: ") + utilities::vectorToString(utilities::mapKeys(_dtype)));
        }
    }

    void checkDialectAndOperation(const std::string& dialect, const std::string& operation) {
        if (_op_type.find(dialect) == _op_type.end()) {
            throw std::runtime_error(std::string("Dialect must select from: ") + utilities::vectorToString(utilities::mapKeys(_op_type)));
        }
        if (_op_type.at(dialect).find(operation) == _op_type.at(dialect).end()) {
            throw std::runtime_error(std::string("Operation must select from: ") + utilities::vectorToString(utilities::mapKeys(_op_type.at(dialect))));
        }
    }

    int get_dimension() const {
        return _dimension;
    }
    std::string get_op_name() const {
        return _vname;
    }
    std::vector<Edge*>& get_in() {
        return _in_data;
    }
    std::vector<Edge*> get_out() const {
        return _out_data;
    }
    std::unordered_map<std::string, std::unordered_map<std::string, int>> get_op() const {
        return _op_type;
    }
    std::unordered_map<std::string, int> get_dtype() const {
        return _dtype;
    }
    Graph::vertex_descriptor get_descriptor() const {
        return _descriptor;
    }


    void set_op(std::string dialect, std::string operation) {
        checkDialectAndOperation(dialect,operation);
        _op_type[dialect][operation]=1;
    }    
    void set_dimension(int dimension) {
        _dimension=dimension;
    }
    void set_name(std::string str) {
        _vname = str;
    }
    void set_dtype(std::string dtype) {
        checkdtype(dtype);
        _dtype[dtype]=1;
    }
    void set_input(std::vector<Edge*>& in) {
        _in_data = in;
    }
    void update_input(Edge* in) {
        _in_data.push_back(in);
    }
    void update_output(Edge* out) {
        _out_data.push_back(out);
    }
    void set_output(std::vector<Edge*>& out) {
        _out_data = out;
    }
    void set_descriptor(Graph::vertex_descriptor& descriptor){
        _descriptor=descriptor;
    }
    void set(std::string str, std::string dialect, std::string operation, std::vector<Edge*>& in, std::vector<Edge*>& out, std::string dtype, int dimension)  {
        checkDialectAndOperation(dialect,operation);
        set_name(str); set_input(in); set_output(out); set_op(dialect, operation); set_dtype(dtype); set_dimension(dimension);
    }
};

class Edge {
protected:
    std::string _ename = "";  // Name of the edge
    Vertex* _in_op;               // Start vertex
    std::vector<Vertex*> _out_op; // End vertices               // Data type of the edge
    std::unordered_map<std::string, int> _dtype {
        {{"None",0}, {"Index",0}, {"I32",0}, {"I64",0}, {"F32",0}, {"F64",0}, {"I32Index",0}, {"I64Index",0} }
        };
    int _dimension;
public:

    // Parametrized constructor
    Edge() {}
    Edge(std::string name, Vertex* in, const std::vector<Vertex*>& out, std::string dtype, int dimension) 
        : _ename(name), _in_op(in), _out_op(out), _dimension(dimension) {
            checkdtype(dtype);
            _dtype[dtype]=1;
        }
    ~Edge() {}  // Virtual destructor


    void checkdtype(const std::string& dtype) {
        if (_dtype.find(dtype) == _dtype.end()) {
            throw std::runtime_error(std::string("Edge Dtype must select from: ") + utilities::vectorToString(utilities::mapKeys(_dtype)));
        }
    }


    // Getter functions
    int get_dimension() const {
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
    void setEdgeName(const std::string& name) {
        _ename = name;
    }
    void set_dimension(int dimension) {
        _dimension=dimension;
    }
    void setInOp(Vertex* in) {
        _in_op = in;
    }

    void setOutOp(std::vector<Vertex*>& out) {
        _out_op = out;
    }

    void updateOutOp(Vertex* op, Graph* g) {
        if (_out_op[0]->get_op_name()=="Sink"){
            _out_op.erase(_out_op.begin());
            for (size_t i = 0; i < (*g)[1]->get_in().size(); ++i) {
                Edge* inEdge=(*g)[1]->get_in()[i];
                if (_ename==inEdge->getEdgeName()){
                    (*g)[1]->get_in().erase((*g)[1]->get_in().begin()+i);
                    break;
                }
            }
        }
        _out_op.push_back(op);
    }

    void setDtype(std::string dtype) {
        checkdtype(dtype);
        _dtype[dtype]=1;
    }

    void set(std::string name, Vertex* in, std::vector<Vertex*>& out, std::string dtype, int dimension) {
        setEdgeName(name); setInOp(in); setOutOp(out); setDtype(dtype);set_dimension(dimension);
    }
};



class Boost_Graph {
private:
    Graph _g;
    std::vector<Edge*> _edge_list;
    std::unordered_map<std::string, std::unordered_map<std::string,std::vector<Vertex*>>> _vertex_list={
            {"Arith", {{"None",{}}, {"Addi",{}}, {"Muli",{}}, {"Addf",{}}, {"Mulf",{}}, {"Constant",{}}, {"IndexCast",{}}}},
            {"SS", {{"None",{}}, {"Source",{}}, {"Sink",{}}}},
            {"Memref", {{"None",{}}, {"Load",{}}, {"Store",{}}}},
            {"Func", {{"None",{}}, {"Func",{}}, {"Return",{}}}},
            {"Scf", {{"None",{}}, {"For",{}}, {"If",{}}}}
            };
    std::map<std::string, Vertex*> _name2Vertex;
    std::map<std::string, Edge*> _name2Edge;


public:
    Boost_Graph(Graph g) 
        : _g(g){}
    
    ~Boost_Graph() {}  // Virtual destructor

    void printGraph(std::ostream& os = std::cout) {
        // Print vertices
        os << "Vertices:" << std::endl;
        auto vertexPair = boost::vertices(_g);
        for (auto vIt = vertexPair.first; vIt != vertexPair.second; ++vIt) {
            Vertex* v = _g[*vIt];
            if (v) {
                os << "Name: " << v->get_op_name() << std::endl;
                os << "Dtype: " << utilities::findPositionOfOne(v->get_dtype()) << std::endl; // Assuming there's only one dtype
                os << "Dimension: " << v->get_dimension() << std::endl;
                os << "Input Edges: ";
                for (Edge* inEdge : v->get_in()) {
                    os << inEdge->getEdgeName() << " ";
                }
                os << std::endl;
                os << "Output Edges: ";
                for (Edge* outEdge : v->get_out()) {
                    os << outEdge->getEdgeName() << " ";
                }
                os << std::endl;
                os << std::endl;
            }
        }

        // Print edges
        os << "\nEdges:" << std::endl;
        auto edgePair = boost::edges(_g);
        for (auto eIt = edgePair.first; eIt != edgePair.second; ++eIt) {
            Edge* e = _g[*eIt];
            Vertex* srcVertex = _g[boost::source(*eIt, _g)];
            Vertex* tgtVertex = _g[boost::target(*eIt, _g)];

            // if (e && srcVertex && tgtVertex) {
            os << "Edge Name: " << e->getEdgeName() << std::endl;
            os << "Dtype: " << utilities::findPositionOfOne(e->getDtype()) << std::endl; // Assuming there's only one dtype
            os << "Dimension: " << e->get_dimension() << std::endl;
            os << "Source Vertex: " << srcVertex->get_op_name() << std::endl;
            os << "Target Vertices: ";
            for (Vertex* tgt : e->getOutOp()) {
                os << tgt->get_op_name() << " ";
            }
            os << std::endl;
            os << std::endl;
            // }
        }
        os << "\n" << std::endl;
    }

    bool vertex_exists(std::string s) const {
        return _name2Vertex.find(s) != _name2Vertex.end();
    }

    bool edge_exists(std::string s) const {
        // std::cout<<s<<std::endl;
        // utilities::print(_name2Edge);
        // std::cout<<std::endl;
        // std::cout<<std::endl;
        return _name2Edge.find(s) != _name2Edge.end();
    }

    bool edge_exists(Edge* e) const {
        return std::find(_edge_list.begin(), _edge_list.end(), e) != _edge_list.end();
    }

    void add_or_update_vertex(Vertex* v) {
        if (!vertex_exists(v->get_op_name())) {
            Graph::vertex_descriptor desc = boost::add_vertex(v,_g);
            v->set_descriptor(desc);
            _name2Vertex[v->get_op_name()]=v;
            // _vertex_to_descriptor_map[v] = desc;
            _g[desc] = v;
            std::pair<std::string, std::string> position= utilities::findPositionOfOne(v->get_op());
            _vertex_list[position.first][position.second].push_back(v);
        }
    }

    void add_edge_to_graph(Edge* e) {
        if (!edge_exists(e->getEdgeName())) {
            Vertex* src = e->getInOp();
            std::vector<Vertex*> tgt = e->getOutOp();

            // Check and possibly add the source vertex.
            add_or_update_vertex(src);

            // Check and possibly add the target vertices.
            for (auto& v : tgt) {
                add_or_update_vertex(v);
                boost::add_edge(src->get_descriptor(), v->get_descriptor(), e, _g);
            }
            _edge_list.push_back(e);
            _name2Edge[e->getEdgeName()]=e;
        }
    }



    void update_edge_to_graph(std::string ename, Vertex* op, size_t edge_index) {
        if (!edge_exists(ename)) {
            throw std::runtime_error(ename + std::string(" does not exist. Try to use add_edge_to_graph function"));
        }
        else{            
            // std::cout<<op->get_in()[1]<<std::endl;
            // std::cout<<fake_edge<<std::endl;
            Edge* edge=_name2Edge[ename];
            // Vertex* src=edge->getInOp();
            add_or_update_vertex(op);


            edge->updateOutOp(op, &_g);

            // boost::add_edge(src->get_descriptor(), op->get_descriptor(), edge, _g);
            // op->update_input(edge);
            op->get_in()[edge_index]=edge;
        }
    }


    Graph& getGraph() {
        return _g;
    }

    const std::vector<Edge*>& getEdgeList() const {
        return _edge_list;
    }

    const std::unordered_map<std::string, std::unordered_map<std::string,std::vector<Vertex*>>>& getVertexList() const {
        return _vertex_list;
    }

    // std::map<std::string, Vertex*> _name2Vertex;
    const std::map<std::string, Vertex*>& getname2Vertex() const {
        return _name2Vertex;
    }

    const std::map<std::string, Edge*>& getname2Edge() const {
        return _name2Edge;
    }
};



#endif