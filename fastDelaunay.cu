#include "math_base.cuh"
#include "utils.cuh"
#include "device_launch_parameters.h"
#include "fastDelaunay.h"

__device__ int getNextAdjTri(const int3& tri, const int3& adjTri, int top)
{
	if (tri.x == top) { return adjTri.x; }
	else if (tri.y == top) { return adjTri.y; }
	else { return adjTri.z; }
}

__global__ void countEdges(int3* tris, int* counts, int* firstTris, int numTriangles)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numTriangles) return;
	int3 t = tris[tid];
	atomicAdd(&counts[t.x], 1);
	atomicAdd(&counts[t.y], 1);
	atomicAdd(&counts[t.z], 1);
	firstTris[t.x] = tid;
	firstTris[t.y] = tid;
	firstTris[t.z] = tid;
}

__global__ void specialPoints(int3* tris, int3* adjTris, int* counts, int* firstTris, int numTriangles)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numTriangles) return;
	int3 t = tris[tid];
	int3 at = adjTris[tid];
	if (at.x == -1)
	{
		firstTris[t.y] = tid;
		atomicAdd(&counts[t.y], 1);
	}
	if (at.y == -1)
	{
		firstTris[t.z] = tid;
		atomicAdd(&counts[t.z], 1);
	}
	if (at.z == -1)
	{
		firstTris[t.x] = tid;
		atomicAdd(&counts[t.x], 1);
	}
}
__global__ void formEdges(int3* tris, int3* adjTris, int3* edgeIds1, int3* edgeIds2, int* heads, int* firstTris, int* froms, int* tos, int numPoints)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numPoints) return;

	int start = firstTris[tid];
	int offset = heads[tid];
	int startOf = offset;
	int3 t = tris[start];
	int id = getIdInt3(t, tid);
	tos[offset] = getInt3(&t, (id + 1) % 3);
	froms[offset] = tid;
	getInt3(&edgeIds1[start], (id + 2) % 3) = offset;
	offset++;
	getInt3(&edgeIds2[start], (id + 1) % 3) = offset;
	
	int next = getInt3(adjTris + start, (id + 1) % 3);
	int pre;
	while (next != -1 && next != start)
	{
		t = tris[next];
		id = getIdInt3(t, tid);
		tos[offset] = getInt3(&t, (id + 1) % 3);
		froms[offset] = tid;
		getInt3(&edgeIds1[next], (id + 2) % 3) = offset;
		offset++;
		
		pre = next;
		next = getInt3(adjTris + next, (id + 1) % 3);
		if (next == start) {
			getInt3(&edgeIds2[pre], (id + 1) % 3) = startOf;
		}
		else {
			getInt3(&edgeIds2[pre], (id + 1) % 3) = offset;
		}
	}
	if (next == -1)
	{
		tos[offset] = getInt3(&t, (id + 2) % 3);
		froms[offset] = tid;
		//getInt3(&edgeIds2[pre], (id + 1) % 3) = offset;
	}
}

__global__ void formOthers(int3* tris, int3* adjTris, int3* edgeIds1, int3* edgeIds2, int* others, int numTriangles)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numTriangles) return;
	//int3 t = tris[tid];
	//int3 at = adjTris[tid];
	//int3 ids = edgeIds[tid];
	//int3 ot;
	//int p;
	//
	//if (at.x != -1) {
	//	ot = tris[at.x];
	//	if (ot.x != t.y && ot.x != t.z)
	//	{
	//		p = 0;
	//	}
	//	else if (ot.y != t.y && ot.y != t.z)
	//	{
	//		p = 1;
	//	}
	//	else
	//	{
	//		p = 2;
	//	}
	//	others[ids.x] = getInt3(edgeIds + at.x, p);
	//}
	//else {
	//	others[ids.x] = -1;
	//}
	//
	//if (at.y != -1) {
	//	ot = tris[at.y];
	//	if (ot.x != t.x && ot.x != t.z)
	//	{
	//		p = 0;
	//	}
	//	else if (ot.y != t.x && ot.y != t.z)
	//	{
	//		p = 1;
	//	}
	//	else
	//	{
	//		p = 2;
	//	}
	//	others[ids.y] = getInt3(edgeIds + at.y, p);
	//}
	//else {
	//	others[ids.y] = -1;
	//}
	//
	//if (at.z != -1) {
	//	ot = tris[at.z];
	//	if (ot.x != t.y && ot.x != t.x)
	//	{
	//		p = 0;
	//	}
	//	else if (ot.y != t.y && ot.y != t.x)
	//	{
	//		p = 1;
	//	}
	//	else
	//	{
	//		p = 2;
	//	}
	//	others[ids.z] = getInt3(edgeIds + at.z, p);
	//}
	//else {
	//	others[ids.z] = -1;
	//}

	int3 e1 = edgeIds1[tid];
	int3 e2 = edgeIds2[tid];
	others[e1.x] = e2.x;
	others[e1.y] = e2.y;
	others[e1.z] = e2.z;
	others[e2.x] = e1.x;
	others[e2.y] = e1.y;
	others[e2.z] = e1.z;
}
double makeEdges(void* cubTemp, size_t cubBytes, int3* tris, int3* adjTris, int* heads, int* froms, int* tos, int* others, int numTriangles, int numPoints, int& numEdges)
{
	int* firstTris;
	utils::malloc(firstTris, numPoints);
	int3* edgeIds1, *edgeIds2;
	utils::malloc(edgeIds1, numTriangles);
	utils::malloc(edgeIds2, numTriangles);
	printf("%d %d\n", numPoints, numTriangles);
	cudaEvent_t start, stop;
	float elapsedTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	utils::memset(heads, numPoints + 1);
	utils::kernel(countEdges, numTriangles, 128, tris, heads, firstTris, numTriangles);
	utils::kernel(specialPoints, numTriangles, 128, tris, adjTris, heads, firstTris, numTriangles);
	utils::exlusiveScan(cubTemp, cubBytes, heads, heads, numPoints + 1);
	numEdges = utils::getValue(heads, numPoints);
	
	utils::kernel(formEdges, numPoints, 128, tris, adjTris, edgeIds1, edgeIds2, heads, firstTris, froms, tos, numPoints);
	utils::kernel(formOthers, numTriangles, 128, tris, adjTris, edgeIds1, edgeIds2, others, numTriangles);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	utils::release(firstTris, numPoints);
	utils::release(edgeIds1, numTriangles);
	utils::release(edgeIds2, numTriangles);
	printf("make edges %f\n", elapsedTime);
	return elapsedTime;
}

__global__ void makePointList(Point2d* points, int* tos, Point2d* toPointList, int numEdges)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numEdges) return;

	toPointList[tid] = points[tos[tid]];
}
__global__ void testNeighborInCircle(Point2d* points, Point2d* toPointList, int* heads, int* froms, int* tos, int* canFlip, int numEdges)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numEdges) return;

	int from = froms[tid];
	Point2d fp = points[from];
	int to = tos[tid];
	Point2d tp = toPointList[tid];

	int L = heads[from];
	int R = heads[from + 1];
	int prv, nxt;
	Point2d np, pp;
	nxt = (tid == R - 1) ? L : tid + 1;
	np = toPointList[nxt];
	prv = (tid == L) ? R - 1 : tid - 1;
	pp = toPointList[prv];

	if (inCircle(pp, tp, fp, np))
	{
		atomicMax(canFlip + from, to);
	}
}

__global__ void starSplay(Point2d* points, Point2d* toPointList, int* heads, int* froms, int* tos, int* canFlip, int numPoints)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int wid = tid / 32;
	tid = tid % 32;
	if (wid >= numPoints) return;
	if (canFlip[wid] > wid) return;
	if (heads[wid + 1] - heads[wid] > 200 && tid == 0)
	{
		printf("%d %d\n", wid, heads[wid + 1] - heads[wid]);
	}
	//int L = heads[wid], R = heads[wid + 1];
	//int eid = L + tid * 2;
	//Point2d fp = points[wid];
	//while (eid < R)
	//{
	//	Point2d tp = toPointList[eid];
	//	int prv, nxt;
	//	Point2d np, pp;
	//	nxt = (eid == R - 1) ? L : eid + 1;
	//	np = toPointList[nxt];
	//	prv = (eid == L) ? R - 1 : eid - 1;
	//	pp = toPointList[prv];
	//}
}

__device__ void flipStar()
{

}
__global__ void starSplay()
{

}
double mis(int* heads, int* froms, int* tos, int* canFlip, int numEdges)
{
	return 0;
}

struct EdgeInfo
{
	int to;
	int other;
};
#include<vector>

struct EdgeInfo2
{
	int2 faces;
	int4 edges;
	int2 points;
};

int insertEdge(std::vector<EdgeInfo>* edges, int after, int from, int to, int other = -1)
{
	edges[from].push_back(EdgeInfo());
	for (int i = edges[from].size() - 1; i >= after + 1; i--)
	{
		auto e = edges[from][i] = edges[from][i - 1];
		edges[e.to][e.other].other = i;
	}
	edges[from][after] = EdgeInfo{ to, other };
	if (other != -1) {
		edges[to][other].other = after;
	}
	return after;
}
int deleteEdge(std::vector<EdgeInfo>* edges, int idx, int from)
{
	size_t sz = edges[from].size();
	for (int i = idx; i < sz - 1; i++)
	{
		auto e = edges[from][i] = edges[from][i + 1];
		edges[e.to][e.other].other = i;
	}
	edges[from].resize(sz - 1);
	return idx;
}

double flipEdge(std::vector<EdgeInfo>* edges, int idx, int from)
{
	int to = edges[from][idx].to;
	auto ne = (idx == edges[from].size() - 1) ? edges[from][0] : edges[from][idx + 1];
	auto pe = (idx == 0) ? edges[from].back() : edges[from][idx - 1];
	deleteEdge(edges, edges[from][idx].other, to);
	insertEdge(edges, pe.other, pe.to, ne.to);
	insertEdge(edges, ne.other + 1, ne.to, pe.to, pe.other);
	deleteEdge(edges, idx, from);
	return 0;
}
int2 getSideV(std::vector<EdgeInfo2>& edges, int eid)
{
	int2 ret;
	auto e = edges[eid];
	auto ex = edges[e.edges.x];
	auto ez = edges[e.edges.z];
	if (ex.points.x != e.points.y) ret.x = ex.points.x; else ret.x = ex.points.y;
	if (ez.points.x != e.points.x) ret.y = ez.points.x; else ret.y = ez.points.y;
	return ret;
}
int flipEdge(std::vector<EdgeInfo2>& edges, int eid)
{
	auto e = edges[eid];
	int2 sideVId = getSideV(edges, eid);

	EdgeInfo2 newE;
	newE.points = { sideVId.y, sideVId.x };
	newE.faces = e.faces;
	newE.edges = { e.edges.y, e.edges.z, e.edges.w, e.edges.x };
	auto& ex = edges[e.edges.x];
	if (ex.points.x == e.points.y)
	{
		ex.faces.x = e.faces.y;
		ex.edges.x = eid; ex.edges.y = e.edges.w;
	}
	else {
		ex.faces.y = e.faces.y;
		ex.edges.z = eid; ex.edges.w = e.edges.w;
	}
	auto& ey = edges[e.edges.y];
	if (ey.points.x == e.points.x)
	{
		ey.edges.z = e.edges.z; ey.edges.w = eid;
	}
	else {
		ey.edges.x = e.edges.z; ey.edges.y = eid;
	}
	auto& ez = edges[e.edges.z];
	if (ez.points.x == e.points.x)
	{
		ez.faces.x = e.faces.x;
		ez.edges.x = eid; ez.edges.y = e.edges.y;
	}
	else {
		ez.faces.y = e.faces.x;
		ez.edges.z = eid; ez.edges.w = e.edges.y;
	}
	auto& ew = edges[e.edges.w];
	if (ew.points.x == e.points.y)
	{
		ew.edges.z = e.edges.x; ew.edges.w = eid;
	}
	else {
		ew.edges.x = e.edges.x; ew.edges.y = eid;
	}
	edges[eid] = newE;
	return 0;
}
bool tryFlipEdge(Point2d* points, std::vector<EdgeInfo2>& edges, int eid)
{
	auto e = edges[eid];
	if (e.faces.x == -1 || e.faces.y == -1)
	{
		return false;
	}
	int2 sideVId = getSideV(edges, eid);
	if (inCircle(points[sideVId.x], points[e.points.x], points[e.points.y], points[sideVId.y]) > 0)
	{
		flipEdge(edges, eid);
		return true;
	}
	return false;
}
bool testEdge(Point2d* points, std::vector<EdgeInfo2>& edges, int eid)
{
	auto e = edges[eid];
	if (e.faces.x == -1 || e.faces.y == -1)
	{
		return false;
	}
	int2 sideVId = getSideV(edges, eid);
	if (inCircle(points[sideVId.x], points[e.points.x], points[e.points.y], points[sideVId.y]) > 0)
	{
		return true;
	}
	return false;
}
int flipMultiEdges(char* labels, Point2d* points, std::vector<EdgeInfo2>& edges, int eid)
{
	int c = 1;
	flipEdge(edges, eid);
	labels[edges[eid].faces.x] = -1;
	labels[edges[eid].faces.y] = -1;
	int ex = edges[eid].edges.x;
	if (labels[edges[ex].faces.x] != -1 || labels[edges[ex].faces.y] != -1)
	{
		if (tryFlipEdge(points, edges, ex))
		{
			labels[edges[ex].faces.x] = -1;
			labels[edges[ex].faces.y] = -1;
			c++;
		}
	}
	
	int ey = edges[eid].edges.y;
	if (labels[edges[ey].faces.x] != -1 || labels[edges[ey].faces.y] != -1)
	{
		if (tryFlipEdge(points, edges, ey))
		{
			labels[edges[ey].faces.x] = -1;
			labels[edges[ey].faces.y] = -1;
			c++;
		}
	}
	
	int ez = edges[eid].edges.z;
	if (labels[edges[ez].faces.x] != -1 || labels[edges[ez].faces.y] != -1)
	{
		if (tryFlipEdge(points, edges, ez))
		{
			labels[edges[ez].faces.x] = -1;
			labels[edges[ez].faces.y] = -1;
			c++;
		}
	}
	
	int ew = edges[eid].edges.w;
	if (labels[edges[ew].faces.x] != -1 || labels[edges[ew].faces.y] != -1)
	{
		if (tryFlipEdge(points, edges, ew))
		{
			labels[edges[ew].faces.x] = -1;
			labels[edges[ew].faces.y] = -1;
			c++;
		}
	}
	return c;
}
int flipStar(Point2d* points, std::vector<EdgeInfo>* edges, int from, bool period)
{
	bool flag = true;
	int flipCount = 0;
	while (flag) {
		flag = false;
		for (int i = 0; i < edges[from].size(); i++)
		{
			if (!period && (i == 0 || edges[from].size() - 1))
			{
				continue;
			}
			auto ne = (i == edges[from].size() - 1) ? edges[from][0] : edges[from][i + 1];
			auto pe = (i == 0) ? edges[from].back() : edges[from][i - 1];
			//printf("%d %d %d %d\n", from, pe.to, edges[from][i].to, ne.to);
			if (inCircle(points[from], points[pe.to], points[edges[from][i].to], points[ne.to]) > 0)
			{
				flipEdge(edges, i, from);
				flag = true;
				flipCount++;
			}
		}
	}
	return flipCount;
}

void getMis(std::vector<EdgeInfo>* edges, char* isMis, int* random, int numPoints)
{
	for (int i = 0; i < numPoints; i++)
	{
		if (isMis[i] == 0)
		{
			bool flag = true;
			for (auto edge : edges[i])
			{
				if (isMis[edge.to] != -1 && (random[edge.to] > random[i] || (random[edge.to] == random[i] && edge.to > i)))
				{
					flag = false;
					break;
				}
			}
			if (flag)
			{
				isMis[i] = 1;
			}
			else {
				isMis[i] = -1;
			}
		}
	}
}
void getEdgeInfo(int* heads, int* froms, int* tos, int* others, int numPoints, std::vector<EdgeInfo>* edges)
{
	for (int i = 0; i < numPoints; i++)
	{
		for (int j = heads[i]; j < heads[i + 1]; j++)
		{
			if (others[j] != -1) {
				edges[i].emplace_back(EdgeInfo{ tos[j], others[j] - heads[froms[others[j]]] });
				if (tos[others[j]] != i)
				{
					printf("wrong info %d %d %d %d\n", i, tos[j], others[j], tos[others[j]]);
				}
			}
			else {
				edges[i].emplace_back(EdgeInfo{ tos[j], -1 });
			}
		}
	}
}
void testEdgeInfo(std::vector<EdgeInfo>* edges, int numPoints)
{
	for (int from = 0; from < numPoints - 4; from++)
	{
		for (int idx = 0; idx < edges[from].size(); idx++)
		{
			int to = edges[from][idx].to;
			auto ne = (idx == edges[from].size() - 1) ? edges[from][0] : edges[from][idx + 1];
			auto pe = (idx == 0) ? edges[from].back() : edges[from][idx - 1];

			int o = edges[from][idx].other;
			auto oe = edges[to][edges[from][idx].other];
			if (oe.to != from)
			{
				printf("%d %d %d %d\n", from, to, o, oe.to);
				printf("wrong other\n");
				return;
			}
			auto one = (o == edges[to].size() - 1) ? edges[to][0] : edges[to][o + 1];
			auto ope = (o == 0) ? edges[to].back() : edges[to][o - 1];
			//if (one.to != pe.to)
			//{
			//	printf("wrong nei 1\n");
			//}
			//if (ope.to != ne.to)
			//{
			//	printf("wrong nei 2\n");
			//}
		}
	}
}
void markMis(Point2d* points, char* isMis, bool* isPeriod, std::vector<EdgeInfo>* edges, int numPoints)
{
	for (int from = 0; from < numPoints; from++)
	{
		bool flag = false;
		for (int i = 0; i < edges[from].size(); i++)
		{
			if (!isPeriod[from] && (i == 0 || edges[from].size() - 1))
			{
				continue;
			}
			auto ne = (i == edges[from].size() - 1) ? edges[from][0] : edges[from][i + 1];
			auto pe = (i == 0) ? edges[from].back() : edges[from][i - 1];
			//printf("%d %d %d %d\n", from, pe.to, edges[from][i].to, ne.to);
			
			if (inCircle(points[from], points[pe.to], points[edges[from][i].to], points[ne.to]) > 0)
			{
				flag = true;
				break;
			}
		}
		if (!flag)
		{
			isMis[from] = -1;
		}
	}
}
double testFastDelaunay(Point2d* points, int* heads, int* froms, int* tos, int* others, int numTriangles, int numPoints)
{
	bool* isPeriod = new bool[numPoints];
	char* isMis = new char[numPoints];
	for (int i = 0; i < numPoints-4; i++)
	{
		isPeriod[i] = true;
	}
	for (int i = numPoints - 4; i < numPoints; i++)
	{
		isPeriod[i] = false;
	}
	std::vector<EdgeInfo>* edges = new std::vector<EdgeInfo>[numPoints];
	printf("get edge\n");
	getEdgeInfo(heads, froms, tos, others, numPoints, edges);
	testEdgeInfo(edges, numPoints);
	
	printf("get end\n");

	int* random = new int[numPoints];
	int c = 1313131313;
	for (int i = 0; i < numPoints; i++)
	{
		c = c * 13 + 19;
		random[i] = c;
	}
	int tt = 50;
	while (tt--) {
		memset(isMis, 0, sizeof(bool) * numPoints);

		markMis(points, isMis, isPeriod, edges, numPoints);
		getMis(edges, isMis, random, numPoints);
		int countMis = 0;
		int maxFlip = 0, sumFlip = 0;
		for (int i = 0; i < numPoints; i++)
		{
			//if (isMis[i] == 1)
			//{
				int flipCount = flipStar(points, edges, i, isPeriod[i]);
				countMis++;
				sumFlip += flipCount;
				if (flipCount > maxFlip) maxFlip = flipCount;
			//}
		}
		printf("%d %d %d\n", countMis, sumFlip, maxFlip);
	}
	delete[] isMis;
	delete[] isPeriod;
	delete[] edges;

	return 0;
}
#include <map>
int makeEdge(std::map<int, int>* tempEdges, int x, int y, std::vector<EdgeInfo2>& edges)
{
	auto it = tempEdges[y].find(x);
	if (it == tempEdges[y].end())
	{
		tempEdges[x][y] = edges.size();
		EdgeInfo2 info;
		info.points.x = x; info.points.y = y;
		info.faces.x = -1; info.faces.y = -1;
		edges.push_back(info);
		return edges.size() - 1;
	}
	else {
		return -tempEdges[y][x];
	}
}
void makeEdgeInfo2(int3* tris, int numTriangles, int numPoints, std::vector<EdgeInfo2>& edges)
{
	std::map<int, int>* tempEdges = new std::map<int, int>[numPoints];
	for (int i = 0; i < numTriangles; i++)
	{
		int3 t = tris[i];
		int ez = makeEdge(tempEdges, t.x, t.y, edges);
		int ex = makeEdge(tempEdges, t.y, t.z, edges);
		int ey = makeEdge(tempEdges, t.z, t.x, edges);
		if (ez > 0)
		{
			edges[ez].faces.x = i;
			edges[ez].edges.x = abs(ex);
			edges[ez].edges.y = abs(ey);
		}
		else {
			edges[-ez].faces.y = i;
			edges[-ez].edges.z = abs(ex);
			edges[-ez].edges.w = abs(ey);
		}

		if (ex > 0)
		{
			edges[ex].faces.x = i;
			edges[ex].edges.x = abs(ey);
			edges[ex].edges.y = abs(ez);
		}
		else {
			edges[-ex].faces.y = i;
			edges[-ex].edges.z = abs(ey);
			edges[-ex].edges.w = abs(ez);
		}

		if (ey > 0)
		{
			edges[ey].faces.x = i;
			edges[ey].edges.x = abs(ez);
			edges[ey].edges.y = abs(ex);
		}
		else {
			edges[-ey].faces.y = i;
			edges[-ey].edges.z = abs(ez);
			edges[-ey].edges.w = abs(ex);
		}
	}
	delete[] tempEdges;
}
double tryFastDelaunay(void* cubTemp, size_t cubBytes, Point2d* points, int3* tris, int3* adjTris, int numPoints, int numTriangles)
{
	int3* h_tris = new int3[numTriangles];
	utils::memcpy(h_tris, tris, numTriangles, cudaMemcpyDeviceToHost);
	Point2d* h_points = new Point2d[numPoints];
	utils::memcpy(h_points, points, numPoints, cudaMemcpyDeviceToHost);
	std::vector<EdgeInfo2> edges;
	edges.reserve(numTriangles * 3 / 2 + 40);
	makeEdgeInfo2(h_tris, numTriangles, numPoints, edges);
	printf("edges.size %lld\n", edges.size());
	char* isMis = new char[edges.size()];
	char* notD = new char[edges.size()];
	char* labels = new char[edges.size()];
	int tt = 50;
	while (tt--) {
		memset(isMis, 0, sizeof(char) * edges.size());
		memset(notD, 0, sizeof(char) * edges.size());

		for (int i = 0; i < edges.size(); i++)
		{
			notD[i] = testEdge(h_points, edges, i);
		}
		for (int i = 0; i < edges.size(); i++)
		{
			if (notD[i])
			{
				auto e = edges[i];
				if (notD[e.edges.x])
				{
					auto sv = getSideV(edges, e.edges.x);
					int p1;
					if (sv.x != e.points.x && sv.x != e.points.y)
					{
						p1 = sv.x;
					}
					else {
						p1 = sv.y;
					}
					sv = getSideV(edges, i);
					int p2 = sv.y;
					if (inCircle(h_points[p1], h_points[e.points.x], h_points[e.points.y], h_points[p2]) < 0)
					{
						printf("error x\n");
					}
				}

				if (notD[e.edges.y])
				{
					auto sv = getSideV(edges, e.edges.y);
					int p1;
					if (sv.x != e.points.x && sv.x != e.points.y)
					{
						p1 = sv.x;
					}
					else {
						p1 = sv.y;
					}
					sv = getSideV(edges, i);
					int p2 = sv.y;
					if (inCircle(h_points[p1], h_points[e.points.x], h_points[e.points.y], h_points[p2]) < 0)
					{
						printf("error y\n");
					}
				}

				if (notD[e.edges.z])
				{
					auto sv = getSideV(edges, e.edges.z);
					int p1;
					if (sv.x != e.points.x && sv.x != e.points.y)
					{
						p1 = sv.x;
					}
					else {
						p1 = sv.y;
					}
					sv = getSideV(edges, i);
					int p2 = sv.x;
					if (inCircle(h_points[e.points.x], h_points[p1], h_points[e.points.y], h_points[p2]) < 0)
					{
						printf("error z\n");
					}
				}

				if (notD[e.edges.w])
				{
					auto sv = getSideV(edges, e.edges.w);
					int p1;
					if (sv.x != e.points.x && sv.x != e.points.y)
					{
						p1 = sv.x;
					}
					else {
						p1 = sv.y;
					}
					sv = getSideV(edges, i);
					int p2 = sv.x;
					if (inCircle(h_points[e.points.x], h_points[p1], h_points[e.points.y], h_points[p2]) < 0)
					{
						printf("error w\n");
					}
				}
			}
		}
		for (int i = 0; i < edges.size(); i++)
		{
			if (notD[i])
			{
				isMis[i] = 1;
				if (notD[edges[i].edges.x] != 0 && edges[i].edges.x > i)
				{
					isMis[i] = 0;
				}
				if (notD[edges[i].edges.y] != 0 && edges[i].edges.y > i)
				{
					isMis[i] = 0;
				}
				if (notD[edges[i].edges.z] != 0 && edges[i].edges.z > i)
				{
					isMis[i] = 0;
				}
				if (notD[edges[i].edges.w] != 0 && edges[i].edges.w > i)
				{
					isMis[i] = 0;
				}
			}
		}
		
		int countMis = 0;
		int sumFlip = 0;
		memset(labels, 0, sizeof(char) * edges.size());
		for (int i = 0; i < edges.size(); i++)
		{
			if (isMis[i] == 1)
			{
				labels[edges[i].faces.x] = -1;
				labels[edges[i].faces.y] = -1;
			}
		}
		for (int i = 0; i < edges.size(); i++)
		{
			//if (isMis[i] == 1)
			//{
			if (isMis[i] == 1) {
				sumFlip += flipMultiEdges(labels, h_points, edges, i);
				countMis++;
			}
			
			//}
		}
		printf("%d %d\n", countMis, sumFlip);
	}
	return 0;
}