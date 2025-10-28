#include "defs.h"
#include "utils.h"
#include <cassert>
#include <cstdlib>
#include <queue>

int main(int argc, char** argv) {
    assert(argc == 4);
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    int k = atoi(argv[3]);

    float* foo;
    int n = read_vecs<float>(argv[1], foo, DIM);
    point_t<float>* points = reinterpret_cast<point_t<float>*>(foo);
    assert(n > 0);

    int* neighbors = new int[n * k];
    for (int i = 0; i < n; i++) {
        std::priority_queue<std::pair<float, int>> pq;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            pq.emplace(dis(points[i], points[j]), j);
            if (pq.size() > k) {
                pq.pop();
            }
        }
        for (int j = k - 1; j >= 0; --j) {
            neighbors[i * k + j] = pq.top().second;
            pq.pop();
        }
        printf("%d / %d\r", i + 1, n);
    }

    write_vecs(output_file, neighbors, n, k);

    delete[] points;
    return 0;
}