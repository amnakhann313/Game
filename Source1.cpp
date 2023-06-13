#include <iostream>
#include <conio.h>
#include <queue>

using namespace std;

template <typename T>
class Vector {
private:
    T* data;
    size_t size;
    size_t capacity;

public:
    // Constructor
    Vector() : data(nullptr), size(0), capacity(0) {}

    // Destructor
    ~Vector() {
        delete[] data;
    }

    // Copy constructor
    Vector(const Vector& other) : data(nullptr), size(other.size), capacity(other.size) {
        data = new T[capacity];
        for (size_t i = 0; i < size; ++i) {
            data[i] = other.data[i];
        }
    }

    // Assignment operator
    Vector& operator=(const Vector& other) {
        if (this != &other) {
            delete[] data;
            size = other.size;
            capacity = other.size;
            data = new T[capacity];
            for (size_t i = 0; i < size; ++i) {
                data[i] = other.data[i];
            }
        }
        return *this;
    }

    // Push an element to the back of the vector
    void push_back(const T& element) {
        if (size >= capacity) {
            capacity = capacity == 0 ? 1 : capacity * 2;
            T* newData = new T[capacity];
            for (size_t i = 0; i < size; ++i) {
                newData[i] = data[i];
            }
            delete[] data;
            data = newData;
        }
        data[size++] = element;
    }

    // Access an element at the specified index
    T& operator[](size_t index) {
        return data[index];
    }

    // Get the size of the vector
    size_t getSize() const {
        return size;
    }
};

// Define the map size
const int SIZEE = 10;
vector<int> currentInventory;
vector< pair<int, int>> inventory;
bool GameEnd = false;
const int INF = numeric_limits<int>::max();

// Define the node structure for the graph
struct Node {
    int id;
    vector< pair<int, int>>neighbors;
    int reward_score; // score of a particular reward you obtained
    int count; // to avoid duplicate id nodes (maintain count of the id)
    Node* next;
    Node* left;
    Node* right;
    int height;

    // Default constructor
    Node() {
        id = 0;
        reward_score = 0;
        count = 0;
        next = nullptr;
        left = nullptr;
        right = nullptr;
        height = 0;
    }
    // Constructor
    Node(int id, int reward_score) : id(id), reward_score(reward_score), count(1), next(nullptr) {}
};

class AVLTree {
private:

    Node* root;

    // Function to create a new node
    Node* createNode(int id, int reward_score) {
        Node* newNode = new Node();
        newNode->id = id;
        newNode->reward_score = reward_score;
        newNode->count = 0;
        newNode->left = nullptr;
        newNode->right = nullptr;
        newNode->height = 1;
        return newNode;
    }

    // Function to get the height of a node
    int getHeight(Node* node) {
        if (node == nullptr) {
            return 0;
        }
        return node->height;
    }

    // Function to calculate the balance factor of a node
    int getBalanceFactor(Node* node) {
        if (node == nullptr) {
            return 0;
        }
        return getHeight(node->left) - getHeight(node->right);
    }

    // Function to update the height of a node
    void updateHeight(Node* node) {
        int leftHeight = getHeight(node->left);
        int rightHeight = getHeight(node->right);
        node->height = max(leftHeight, rightHeight) + 1;
    }

    // Function to perform a right rotation
    Node* rotateRight(Node* y) {
        Node* x = y->left;
        Node* T2 = x->right;

        x->right = y;
        y->left = T2;

        updateHeight(y);
        updateHeight(x);

        return x;
    }

    // Function to perform a left rotation
    Node* rotateLeft(Node* x) {
        Node* y = x->right;
        Node* T2 = y->left;

        y->left = x;
        x->right = T2;

        updateHeight(x);
        updateHeight(y);

        return y;
    }

    // Function to balance the tree after insertion or deletion
    Node* balance(Node* node, int id) {
        int balanceFactor = getBalanceFactor(node);

        // Left Left Case
        if (balanceFactor > 1 && id < node->left->id) {
            return rotateRight(node);
        }

        // Right Right Case
        if (balanceFactor < -1 && id > node->right->id) {
            return rotateLeft(node);
        }

        // Left Right Case
        if (balanceFactor > 1 && id > node->left->id) {
            node->left = rotateLeft(node->left);
            return rotateRight(node);
        }

        // Right Left Case
        if (balanceFactor < -1 && id < node->right->id) {
            node->right = rotateRight(node->right);
            return rotateLeft(node);
        }

        return node;
    }

    // Function to insert a node into the AVL tree
    Node* insertNode(Node* node, int id, int reward_score) {
        if (node == nullptr) {
            return createNode(id, reward_score);
        }

        if (id < node->id) {
            node->left = insertNode(node->left, id, reward_score);
        }
        else if (id > node->id) {
            node->right = insertNode(node->right, id, reward_score);
        }
        else {
            node->count++;
            return node;
        }

        updateHeight(node);
        node = balance(node, id);

        return node;
    }

    // Function to find the node with the minimum ID in the AVL tree
    Node* findMinNode(Node* node) {
        if (node == nullptr || node->left == nullptr) {
            return node;
        }
        return findMinNode(node->left);
    }

    Node* deleteNode(Node* node, int id, vector< pair<int, int>>& inventory) {
        if (node == nullptr) {
            return node;
        }

        if (id < node->id) {
            node->left = deleteNode(node->left, id, inventory);
        }
        else if (id > node->id) {
            node->right = deleteNode(node->right, id, inventory);
        }
        else {
            if (node->count > 1) {
                node->count--;


            }
            else
            {
                inventory.erase(remove_if(inventory.begin(), inventory.end(),
                    [node](const  pair<int, int>& item) {
                        return item.first == node->reward_score;
                    }), inventory.end());

                if (node->left == nullptr || node->right == nullptr) {
                    Node* temp = node->left ? node->left : node->right;

                    if (temp == nullptr) {
                        temp = node;
                        node = nullptr;
                    }
                    else {
                        *node = *temp;
                    }

                    delete temp;
                }
                else {
                    Node* temp = findMinNode(node->right);
                    node->id = temp->id;
                    node->reward_score = temp->reward_score;
                    node->count = temp->count;
                    node->right = deleteNode(node->right, temp->id, inventory);
                }
            }
        }

        if (node == nullptr) {
            return node;
        }

        updateHeight(node);
        node = balance(node, id);

        return node;
    }


    // Recursive in-order traversal
    void inOrderTraversal(Node* node, vector< pair<int, int>>& inventory) {
        if (node == nullptr) {
            return;
        }

        inOrderTraversal(node->left, inventory);

        if (node->count > 1) {
            inventory.push_back(make_pair(node->reward_score, node->count));
        }
        else {
            inventory.push_back(make_pair(node->reward_score, 1));
        }

        inOrderTraversal(node->right, inventory);
    }



public:
    // Constructor
    AVLTree() : root(nullptr) {}

    // Function to retrieve the inventory in-order
    vector< pair<int, int>> getInOrderTraversal() {
        // vector< pair<int, int>> inventory;
        inventory.clear();
        inOrderTraversal(root, inventory);
        return inventory;
    }


    // Function to insert a score into the AVL tree
    void insertScore(int id, int reward_score) {
        root = insertNode(root, id, reward_score);
    }

    // Function to update the inventory after deleting a node
    void updateInventory(const  vector< pair<int, int>>& inventory) {
        // Clear the current inventory
        currentInventory.clear();

        // Rebuild the inventory based on the updated AVL tree
        for (const auto& item : inventory) {
            int reward_score = item.first;
            int count = item.second;

            for (int i = 0; i < count; i++) {
                currentInventory.push_back(reward_score);
            }
        }
    }

    // Function to delete a score from the AVL tree
    void deleteScore(int id) {

        root = deleteNode(root, id, inventory);
        updateInventory(inventory); // Update the inventory vector
    }

    // Function to check if the AVL tree is empty
    bool isEmpty() const {
        return root == nullptr;
    }
};

// Function to check if a given position is valid on the map
bool isValidPosition(int x, int y) {
    return x >= 0 && x < SIZEE&& y >= 0 && y < SIZEE;
}

int getWeight(char entity) {

    if (entity == '#' || (entity == '@') || (entity == '$') || (entity == '%') || (entity == '&')) {
        return 100; // Obstacle 
    }
    return 1; // Default weight for other entities 
}

// Function to convert the map to a graph using adjacency list with weights
vector<Node> convertToWeightedGraph(const  vector< vector<char>>& map) {
    vector<Node> graph(map.size() * map[0].size());

    int node = 0;
    int rows = map.size();
    int cols = map[0].size();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if ((map[i][j] != '#') && (map[i][j] != '@') && (map[i][j] != '$') && (map[i][j] != '%') && (map[i][j] != '&')) {
                graph[node].id = node;

                //to check boundaries
                if (i > 0 && (map[i - 1][j] != '#') && (map[i - 1][j] != '@') && (map[i - 1][j] != '$') && (map[i - 1][j] != '%') && (map[i - 1][j] != '&')) {
                    int neighbor = node - cols;
                    int weight = (map[i - 1][j] == 'C') ? 1 : getWeight(map[i - 1][j]);
                    graph[node].neighbors.push_back(make_pair(neighbor, weight));

                }
                if ((j > 0) && (map[i][j - 1] != '#') && (map[i][j - 1] != '@') && (map[i][j - 1] != '$') && (map[i][j - 1] != '%') && (map[i][j - 1] != '&')) {
                    int neighbor = node - 1;
                    int weight = (map[i][j - 1] == 'C') ? 1 : getWeight(map[i][j - 1]);
                    graph[node].neighbors.push_back(make_pair(neighbor, weight));

                }
                if ((j < cols - 1) && (map[i][j + 1] != '#') && (map[i][j + 1] != '@') && (map[i][j + 1] != '$') && (map[i][j + 1] != '%') && (map[i][j + 1] != '&')) {
                    int neighbor = node + 1;
                    int weight = (map[i][j + 1] == 'C') ? 1 : getWeight(map[i][j + 1]);
                    graph[node].neighbors.push_back(make_pair(neighbor, weight));

                }
                if ((i < rows - 1) && (map[i + 1][j] != '#') && (map[i + 1][j] != '@') && (map[i + 1][j] != '$') && (map[i + 1][j] != '%') && (map[i + 1][j] != '&')) {
                    int neighbor = node + cols;
                    int weight = (map[i + 1][j] == 'C') ? 1 : getWeight(map[i + 1][j]);
                    graph[node].neighbors.push_back(make_pair(neighbor, weight));

                }
            }
            else {
                graph[node].id = node;
            }
            ++node;
        }
    }

    return graph;
}


// Function to display the graph
void displayGraph(const  vector<Node>& graph) {
    for (const Node& node : graph) {
        cout << "Node " << node.id << ": ";
        for (const pair<int, int>& neighbor : node.neighbors) {
            cout << neighbor.first << " (" << neighbor.second << ") ";
        }
        cout << endl;
    }
}


// Function to initialize the map
vector< vector<char>> initializeMap() {
    vector< vector<char>> map = {
        {'C','#','C','#','C','C','C','C','J','#'},
        {'C', 'C','$','C', 'C', 'C','%','C','C','C'},
        {'#', 'C','C','C','C','C','C','C','#','C' },
        { 'C', 'C', 'C', 'C','C','#','C', 'C', 'C', 'C'},
        {'J','C','C','C','C','$','%','@','C','&'},
        {'C','J','C','C','C','C','P','C','#','&'},
        {'C','C','P','#','C','C','#','C','%','C'},
        {'C','C','C','*','C','C','C','C','W','W'},
        {'C','C','P','C','#','&','$','W','C','%'},
        {'C','C','J','P','W','C','W','P','J','C'}
    };
    return map;
}

// Function to calculate the shortest path using Floyd's algorithm
vector< vector<int>> calculateShortestPaths(const  vector<Node>& graph) {
    int n = graph.size();
    vector< vector<int>> distances(n, vector<int>(n, numeric_limits<int>::max()));

    // Initialize the distance matrix with direct edge weights
    for (int i = 0; i < n; ++i) {
        distances[i][i] = 0;
        for (const pair<int, int>& neighbor : graph[i].neighbors) {
            distances[i][neighbor.first] = neighbor.second;
        }
    }

    // Perform Floyd's algorithm
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (distances[i][k] != numeric_limits<int>::max() &&
                    distances[k][j] != numeric_limits<int>::max() &&
                    distances[i][k] + distances[k][j] < distances[i][j]) {
                    distances[i][j] = distances[i][k] + distances[k][j];
                }
            }
        }
    }

    return distances;
}

pair<int, int> goalposition(vector< vector<char>> map)
{
    for (int i = 0; i < SIZEE; i++) {
        for (int j = 0; j < SIZEE; j++) {
            if (map[i][j] == '*') {
                return make_pair(i, j);
            } 
        }
    }
}
// algo dyjekstra
void displayshortestpath(vector< vector<int>> shortestPaths, const  vector<Node>& graph, vector< vector<char>> map, int i_sc, int j_sc) {
    // Print the shortest paths
    cout << "Shortest Path:" << endl;
    int n = graph.size();
    int m = sqrt(n);
    int index = (i_sc * m) + j_sc;//to calculate 1d array from 2d array 
    int i_goal;
    int j_goal;
    pair<int, int> goal = goalposition(map);
    i_goal = goal.first;
    j_goal = goal.second;

    for (int i = 0; i < n; ++i) {

        for (int j = 0; j < n; ++j) {

            if (j == ((i_goal * m) + j_goal) && i == index)
            {
                if (shortestPaths[i][j] == numeric_limits<int>::max()) {
                }
                else {
                    cout << "(" << i_sc << "," << j_sc << ") -> * (" << shortestPaths[i][j] << ") ";
                    cout << endl << endl;
                    return;
                }
            }
        }

    }


}

vector<int> dijkstra(const  vector<Node>& graph, int i_sc, int j_sc) {
    int numNodes = graph.size();
    vector<int> dist(numNodes, INF); // Distance from source to each node
    vector<bool> visited(numNodes, false); // Mark nodes as visited
    int m = sqrt(numNodes);
    int source = (i_sc * m) + j_sc;
    // Create a priority queue with a custom comparator to store {nodeId, distance} pairs
    auto cmp = [](const  pair<int, int>& a, const  pair<int, int>& b) {
        return a.second > b.second;
    };
    priority_queue< pair<int, int>, vector< pair<int, int>>, decltype(cmp)> pq(cmp);

    dist[source] = 0;
    pq.push(make_pair(source, 0));

    while (!pq.empty()) {
        int node = pq.top().first;
        pq.pop();

        if (visited[node])
            continue;

        visited[node] = true;

        // Traverse neighbors of the current node
        for (const auto& neighbor : graph[node].neighbors) {
            int neighborId = neighbor.first;
            int weight = neighbor.second;

            if (!visited[neighborId] && dist[node] + weight < dist[neighborId]) {
                dist[neighborId] = dist[node] + weight;
                pq.push(make_pair(neighborId, dist[neighborId]));
            }
        }
    }

    return dist;
}

void printDijkstra(vector<int> minimum, int i_sc, int j_sc, vector< vector<char>> map)
{
    int n = map.size();
    int m = sqrt(n);
    int i_goal;
    int j_goal;
    pair<int, int> goal = goalposition(map);
    i_goal = goal.first;
    j_goal = goal.second;
    int index = (i_goal * n) + j_goal;

    // Print shortest distances from the source node to all other nodes
    cout << "Shortest Path: " << endl;
    for (int i = 0; i < minimum.size(); ++i) {

        if (i == index)
        {
            if (minimum[i] == INF) {
            }
            else
            {
                cout << "(" << i_sc << "," << j_sc << ") -> * (" << minimum[i] << ") ";
                cout << endl << endl;
                return;
            }
        }

    }
}
struct Edge {
    int source;
    int destination;
    int weight;

    Edge(int source, int destination, int weight) : source(source), destination(destination), weight(weight) {}
};

std::vector<std::vector<int>> minimumSpanningTree_Prims(const std::vector<Node>& graph) {
    int numVertices = graph.size();
    std::vector<bool> visited(numVertices, false);
    std::vector<std::vector<int>> minimumSpanningTree;

    // Start with the first vertex (0) as the initial node.
    int initialNode = 0;
    visited[initialNode] = true;

    std::vector<Edge> edges;

    // Traverse the neighbors of the initial node
    for (const auto& neighbor : graph[initialNode].neighbors) {
        int destination = neighbor.first;
        int weight = neighbor.second;
        edges.push_back(Edge(initialNode, destination, weight));
    }

    // Sort the edges in non-decreasing order of weight
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.weight < b.weight;
        });

    while (!edges.empty()) {
        Edge currentEdge = edges.front();
        edges.erase(edges.begin());

        int source = currentEdge.source;
        int destination = currentEdge.destination;

        if (visited[source] && !visited[destination]) {
            visited[destination] = true;
            minimumSpanningTree.push_back({ source, destination });

            // Traverse the neighbors of the destination node
            for (const auto& neighbor : graph[destination].neighbors) {
                int nextDestination = neighbor.first;
                int weight = neighbor.second;
                edges.push_back(Edge(destination, nextDestination, weight));
            }

            // Sort the edges again
            std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
                return a.weight < b.weight;
                });
        }
    }

    return minimumSpanningTree;
}

void displayMinimumSpanningTree(const std::vector<std::vector<int>>& minimumSpanningTree) {
    std::cout << "Minimum Spanning Tree:" << std::endl;
    for (const auto& edge : minimumSpanningTree) {
        std::cout << edge[0] << " -- " << edge[1] << std::endl;
    }
}
// Find function for the disjoint set data structure
int find(std::vector<int>& parent, int vertex) {
    if (parent[vertex] == -1)
        return vertex;
    return find(parent, parent[vertex]);
}

// Union function for the disjoint set data structure
void unionSets(std::vector<int>& parent, int x, int y) {
    int xRoot = find(parent, x);
    int yRoot = find(parent, y);
    parent[xRoot] = yRoot;
}

// Compare function for sorting edges
bool compareEdges(const Edge& a, const Edge& b) {
    return a.weight < b.weight;
}

// Kruskal's algorithm for finding the minimum spanning tree
std::vector<std::vector<int>> minimumSpanningTree_Kruskal(const std::vector<Node>& graph) {
    int numVertices = graph.size();

    // Create a disjoint set to track the connected components
    std::vector<int> parent(numVertices, -1);

    // Create a vector to store the edges in the graph
    std::vector<Edge> edges;

    for (int i = 0; i < numVertices; ++i) {
        const Node& node = graph[i];

        for (const auto& neighbor : node.neighbors) {
            edges.push_back(Edge(node.id, neighbor.first, neighbor.second));
        }
    }

    // Sort the edges in non-decreasing order of weight
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.weight < b.weight;
        });

    std::vector<std::vector<int>> minimumSpanningTree;

    for (const auto& edge : edges) {
        int source = edge.source;
        int destination = edge.destination;

        int sourceRoot = find(parent, source);
        int destinationRoot = find(parent, destination);

        if (sourceRoot != destinationRoot) {
            // The edge doesn't form a cycle, include it in the minimum spanning tree
            minimumSpanningTree.push_back({ source, destination });

            // Union the two sets
            unionSets(parent, sourceRoot, destinationRoot);
        }
    }

    return minimumSpanningTree;
}



int displayMenu() {
    int choice = 0;
    cout << endl << endl;
    cout << "=== Menu ===" << endl;
    cout << "1. Play Game" << endl;
    cout << "2. Shortest path using default location (0,0) by Floyd" << endl;
    cout << "3. Shortest path using default location (0,0) by Dijkstra" << endl;
    cout << "4. Shortest path using custom location (i,j) by Floyd" << endl;
    cout << "5. Shortest path using custom location (i,j) by Dijkstra" << endl;
    cout << "6. Calculate the minimum spanning tree of the forest using Prim's algorithm" << endl;
    cout << "7. Calculate the minimum spanning tree of the forest using Kruskal's algorithm" << endl;
    cout << "8. Display Graph" << endl;
    cout << "9. Quit" << endl;
    cout << "=============" << endl;
    cout << "Enter your choice: ";
    cin >> choice;
    cout << endl << endl;
    return choice;
}

void displaymap(vector<vector<char>> map) {
    cout << "Map:" << endl;
    for (const vector<char>& row : map) {
        for (char entity : row) {

            cout << entity << " ";
        }
        cout << endl;
    }
}
#include <Windows.h>
#pragma comment(lib, "Winmm.lib")

class Game {
private:
    AVLTree avlTree;
    vector< vector<char>> map;
    vector<Node> graph;
    vector< vector<int>> shortestPaths;
    int playerX;
    int playerY;

public:
    Game() {
        map = initializeMap();
        graph = convertToWeightedGraph(map);
        shortestPaths = calculateShortestPaths(graph);
        playerX = 0;
        playerY = 0;
    }

    void play() {
        int score = 0;

        while (true) {
            displayMap();
            cout << "Score: " << score << endl;


            displayInventory();

            int newX = playerX;
            int newY = playerY;

            int ch = _getch(); // Capture the input

            if (ch == 224) { // Check for arrow key
                ch = _getch(); // Capture the arrow key

                switch (ch) {
                case 72: // Left arrow key
                    if (isValidPosition(playerX - 1, playerY)) {
                        PlaySound(TEXT("applause_yeah.wav"), NULL, SND_SYNC);
                        newX = playerX - 1;
                    }
                    break;
                case 75: // Down arrow key
                    if (isValidPosition(playerX, playerY - 1)) {
                        PlaySound(TEXT("applause_yeah.wav"), NULL, SND_SYNC);
                        newY = playerY - 1;
                    }
                    break;
                case 80: // Right arrow key
                    if (isValidPosition(playerX + 1, playerY)) {
                        PlaySound(TEXT("applause_yeah.wav"), NULL, SND_SYNC);
                        newX = playerX + 1;
                    }
                    break;
                case 77: // Up arrow key
                    if (isValidPosition(playerX, playerY + 1)) {
                        PlaySound(TEXT("applause_yeah.wav"), NULL, SND_SYNC);
                        newY = playerY + 1;
                    }
                    break;
                default:
                    cout << "Invalid action. Please try again." << endl;
                    continue;
                }
            }
            else {
                cout << "Invalid action. Please try again." << endl;
                continue;
            }

            char entity = map[newX][newY];
            if (entity == '#') {
                cout << "Oops! You hit an obstacle. Change your path!!" << endl;
                continue;

            }

            if (entity == 'J') {
                int reward = 50;
                avlTree.insertScore(int('J'), reward);
                score += reward;
            }

            if (entity == 'W') {
                int reward = 30;
                avlTree.insertScore(int('W'), reward);
                score += reward;
            }

            if (entity == 'P') {
                int reward = 70;
                avlTree.insertScore(int('P'), reward);
                score += reward;
            }

            if (entity == '@') {
                int nodeId = newX * SIZEE + newY;
                int reward = 30;

                avlTree.deleteScore(int('W'));
                score -= reward;

            }

            if (entity == '&') {
                int nodeId = newX * SIZEE + newY;
                int reward = 50;
                avlTree.deleteScore(int('P'));
                score -= reward;
            }

            if (entity == '$') {
                int nodeId = newX * SIZEE + newY;
                int reward = 70;
                avlTree.deleteScore(int('J'));
                score -= reward;
            }
            playerX = newX;
            playerY = newY;

            if (entity == '*') {
                cout << "Congratulations! You reached the goal." << endl;
                break;
            }

            if (entity == '%') {
                cout << "DEATH POINT" << endl;
                break;
            }
        }

        cout << "Game Over" << endl;
        cout << "Final Score: " << score << endl;
        displayInventory();
        cout << endl;
        return;
    }

    void displayMap() {
        cout << "Map:" << endl;
        for (int i = 0; i < SIZEE; i++) {
            for (int j = 0; j < SIZEE; j++) {
                if (i == playerX && j == playerY) {
                    cout << "- ";
                }
                else {
                    cout << map[i][j] << " ";
                }
            }
            cout << endl;
        }
    }

    void displayInventory() {
        cout << "Inventory:" << endl;
        vector< pair<int, int>> inventory = avlTree.getInOrderTraversal();

        for (const auto& item : inventory) {
            cout << "Reward Score: " << item.first << " Count: " << item.second << endl;
        }
        inventory.clear();
        cout << endl << endl;;
    }

};

void handlemenu(int choice, vector< vector<char>> map, vector<Node> graph)
{
    if (choice == 9)
    {
        GameEnd = true;
        return;
    }

    else if (choice == 1)
    {
        Game game;
        game.play();
    }

    else if (choice == 2)
    {
        displaymap(map);
        cout << endl;
        // Calculate the shortest paths using Floyd's algorithm
        vector< vector<int>> shortestPaths = calculateShortestPaths(graph);
        displayshortestpath(shortestPaths, graph, map, 0, 0);
    }

    else if (choice == 3) {
        vector<int> minimum = dijkstra(graph, 0, 0);
        printDijkstra(minimum, 0, 0, map);
    }

    else if (choice == 4)
    {
        displaymap(map);
        cout << endl;
        int i, j;
        cout << "Enter the coordinates to find shortest path to goal: " << endl;
        cout << "i :";
        cin >> i;
        cout << "j :";
        cin >> j;
        cout << endl;

        // Calculate the shortest paths using Floyd's algorithm
        vector< vector<int>> shortestPaths = calculateShortestPaths(graph);
        displayshortestpath(shortestPaths, graph, map, i, j);
    }

    else if (choice == 5) {
        displaymap(map);
        cout << endl;
        int i, j;
        cout << "Enter the coordinates to find shortest path to goal: " << endl;
        cout << "i :";
        cin >> i;
        cout << "j :";
        cin >> j;
        cout << endl;

        vector<int> minimum = dijkstra(graph, i, j);
        printDijkstra(minimum, i, j, map);
    }

    else if (choice == 6) {

        vector<std::vector<int>> mst = minimumSpanningTree_Prims(graph);
        displayMinimumSpanningTree(mst);
    }

    else if (choice == 7) {

        vector<std::vector<int>> mst_k = minimumSpanningTree_Kruskal(graph);
        displayMinimumSpanningTree(mst_k);
    }

    else if (choice == 8) {
        displayGraph(graph);
    }

    else
    {
        cout << "Invalid option" << endl;
        return;
    }

    return;
}

int main() {

    vector< vector<char>> map = initializeMap();
    vector<Node> graph = convertToWeightedGraph(map);
    system("color 4f");
    while (!GameEnd)
    {
        int ch = displayMenu();
        handlemenu(ch, map, graph);
    }
    return 0;
}
