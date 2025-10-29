#include "defs.h"
#include "utils.h"
#include <bits/stdc++.h>

using namespace std;

#define K 10
using pfi = pair<float, int>;

struct vertex_t {
    int neighbors[K];
};

vector<int> search(const point_t<float>& query, const vertex_t* graph, const point_t<float>* points, int n, int k) {
    priority_queue<pfi> results;
    priority_queue<pfi, vector<pfi>, greater<pfi>> candidates;
    vector<int> vis(n, 0);
    vector<int> ans;
    const static int m = min(15 * (int)pow(n, 1.0 / 3), n);

    for (int i = 0; i < m; i++) {
        int entry = rand() % n;
        vis[entry] = 1;
        float dist = dis(query, points[entry]);
        candidates.push({dist, entry});
        results.push({dist, entry});
    }
    while (!candidates.empty()) {
        auto [can_dis, can] = candidates.top();
        candidates.pop();
        if (can < 0 || can >= n) continue;
        if (results.size() >= k && can_dis > results.top().first)
            break;

        for (auto nei : graph[can].neighbors) {
            if (nei < 0 || nei >= n || vis[nei])
                continue;
            vis[nei] = 1;

            float nei_dis = dis(query, points[nei]);
            candidates.push({nei_dis, nei});
            if (results.size() < k)
                results.push({nei_dis, nei});
            else if (nei_dis < results.top().first) {
                results.pop();
                results.push({nei_dis, nei});
            }
        }
    }

    while (results.size() > k)
        results.pop();

    while (!results.empty()) {
        ans.push_back(results.top().second);
        results.pop();
    }
    return ans;
}

int main(int argc, char** argv) {

    assert(argc == 6);
    string input_file = argv[1];
    string graph_file = argv[2];
    string query_file = argv[3];
    string truth_file = argv[4];
    int k = atoi(argv[5]);

    float* foo = nullptr;
    int n = read_vecs(input_file, foo, DIM);
    assert(n > 0);
    point_t<float>* points = reinterpret_cast<point_t<float>*>(foo);

    int* bar = nullptr;
    int graph_n = read_vecs(graph_file, bar, K);
    assert(graph_n == n);
    vertex_t* graph = reinterpret_cast<vertex_t*>(bar);

    int q = read_vecs(query_file, foo, DIM);
    assert(q > 0);
    point_t<float>* queries = reinterpret_cast<point_t<float>*>(foo);

    int* truths = nullptr;
    int truth_q = read_vecs(truth_file, truths, k);
    assert(truth_q == q);

    double tot_time = 0.0;
    long long tot_correct = 0;

    #pragma omp parallel for reduction(+:tot_time) reduction(+:tot_correct)
    for (int i = 0; i < q; ++i) {
        auto start = chrono::high_resolution_clock::now();
        auto result = search(queries[i], graph, points, n, k);
        auto end = chrono::high_resolution_clock::now();
        tot_time += chrono::duration<double, micro>(end - start).count();

        for (int j = 0; j < k; ++j) {
            for (int l = 0; l < result.size(); ++l) {
                if (truths[i * k + j] == result[l]) {
                    tot_correct++;
                    break;
                }
            }
        }
    }
    printf("QPS: %.3f\n", q * 1e6 / tot_time);
    printf("Recall@%d: %.3f%%\n", k, 100.0 * tot_correct / (q * k));

    delete[] graph;
    delete[] points;
    delete[] truths;
    delete[] queries;
    return 0;
}