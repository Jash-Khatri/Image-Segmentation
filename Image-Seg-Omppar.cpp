// Program to implement image segmentation using push-relabel algorithm using graph cuts
/**
Dependencies:
OpenCV, g++, OpenMP.

Constraints:
Input image should be a square grey-scale image.
Input image should be in .png format

compile and Run:
g++ -fopenmp Image-Seg-Omppar.cpp `pkg-config --cflags --libs opencv`
./a.out image_name bseed1 bseed2 bseed3 bseed4 bseed5 fseed1 num_threads

*/
#include <bits/stdc++.h>
#include <math.h>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <omp.h>
#include <sys/time.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <string>

#define KERNEL_CYCLES 1000

using namespace cv;
using namespace std;
  
int totalitr=0;

struct Edge
{
    // To store current flow and capacity of edge
    int flow, capacity;
  
    // An edge u--->v has start vertex as u and end
    // vertex as v.
    int u, v;
  
    Edge(int flow, int capacity, int u, int v)
    {
        this->flow = flow;
        this->capacity = capacity;
        this->u = u;
        this->v = v;
    }
};
  
// Represent a Vertex
struct Vertex
{
    int h, e_flow;
  
    Vertex(int h, int e_flow)
    {
        this->h = h;
        this->e_flow = e_flow;
    }
};
  
// To represent a flow network
class Graph
{
public:
    int V;    // No. of vertices
    vector<Vertex> ver;
    vector<Edge> edge;
  
    // Function to reverse edge
    void updateReverseEdgeFlow(int i, int flow);
  
    Graph(int V);  // Constructor
  
    // function to add an edge to graph
    void addEdge(int u, int v, int w);
  
    // returns maximum flow from s to t
    int getMaxFlow(int s, int t, int thread_count);
};
  
Graph::Graph(int V)
{
    this->V = V;
  
    // all vertices are initialized with 0 height
    // and 0 excess flow
    for (int i = 0; i < V; i++)
        ver.push_back(Vertex(0, 0));
}
  
void Graph::addEdge(int u, int v, int capacity)
{
    // flow is initialized with 0 for all edge
    edge.push_back(Edge(0, capacity, u, v));
}
  
// returns index of overflowing Vertex
int overFlowVertex(vector<Vertex>& ver)
{
    for (int i = 1; i < ver.size() - 1; i++)
       if (ver[i].e_flow > 0)
            return i;
  
    // -1 if no overflowing Vertex
    return -1;
}

// Update reverse flow for flow added on ith Edge
void Graph::updateReverseEdgeFlow(int i, int flow)
{
    int u = edge[i].v, v = edge[i].u;
  
    for (int j = 0; j < edge.size(); j++)
    {
        if (edge[j].v == v && edge[j].u == u)
        {
            edge[j].flow -= flow;
            return;
        }
    }
  
    // adding reverse Edge in residual graph
    Edge e = Edge(0, flow, u, v);
    edge.push_back(e);
} 
  
// OpenMP parallel function for finding max-flow, min-cut of graph
int Graph::getMaxFlow(int s, int t, int thread_count)
{
    // Making h of source Vertex equal to no. of vertices
    // Height of other vertices is 0.
    ver[s].h = ver.size();			// h(s) <- |V|
    int TotalExcess = 0;

	// computing the preflow
    for (int i = 0; i < edge.size(); i++)		// \forall (s, u) \in E do
    {
        // If current edge goes from source
        if (edge[i].u == s)
        {
            // Flow is equal to capacity
            edge[i].flow = edge[i].capacity;		// c_f (s, u) <- c_f (s, u) - c_{su}
	    TotalExcess += edge[i].capacity;		// ExcessTotal <- ExcessTotal + c_{su}
  
            // Initialize excess flow for adjacent v
            ver[edge[i].v].e_flow += edge[i].flow;	// e(u) <- c_{su}
  
            // Add an edge from v to s in residual graph with
            // capacity equal to 0
            edge.push_back(Edge(-edge[i].flow, 0, edge[i].v, s));		// c_f (u, s) <- c_f (u, s) + c_{su}
        }
    }

	int numofver;
	numofver = ver.size();

    // Loop until the convergence criteria is met
    while( ( ver[s].e_flow + ver[t].e_flow ) < TotalExcess )	
    {
	
    		//break up vertices in chunks. Each thread take responsibility for one particular chunk
    		#pragma omp parallel num_threads(thread_count)
    		{
        		int startVx, endVx;
        		if (numofver < omp_get_num_threads()) {
	            		startVx = omp_get_thread_num();
	            		endVx = omp_get_thread_num() + 1;
        		} 
			else {
            			int gap = numofver / omp_get_num_threads();
            			startVx = omp_get_thread_num() * gap;
            			endVx = startVx + gap; // exclusive
            			if (omp_get_thread_num() == (omp_get_num_threads() - 1)) {
                			endVx = numofver;
            			}
        		}

			if(totalitr == 0)
				cout << startVx << " " << endVx << "\n";

			// check for vertex with excess in the assigned chunk to a thread
			int hasexcess = -1;
    			for (int i = startVx; i < endVx ; i++){
				if ((i == s) || (i == t)) continue;
				if (ver[i].e_flow > 0){
					hasexcess = i;
					break;
				}	
         	 	  		
			}
		
			#pragma omp barrier					

			// ****Push_Relabel_Kernel Begins ****

		      // Execute the kernel for the kernel cycles number of times 
		      int cycles=KERNEL_CYCLES;			// cycle = KERNEL_CYCLES
		      
		      while( cycles > 0 )			// while cycle > 0 do
		      {
			   // if excess is +ve then continue
			   if (hasexcess != -1 )			// if e(u) > 0
    			   {		
   		   	 	int u = hasexcess;
				// if the min-height neighbour is present then Push the excess flow to the min-height neighbour. Else relabel the current vertex        	    		
				
				// Initialize minimum height of an adjacent vertex
    				int mh = INT_MAX;				// h' <- INF
				int e_dash = ver[u].e_flow;			// e' <- e(u)
				int v_dash;

				int flag = 0;
				for (int i = 0; i < edge.size(); i++)			// \forall (u, v) \in E
    				{
        				// Checks u of current edge is same as given
        				// overflowing vertex
        				if (edge[i].u == u)
        				{
            					// if flow is equal to capacity then no push
            					// is possible
            					if (edge[i].flow == edge[i].capacity)
                					continue;

						// Update minimum height
            					if (ver[edge[i].v].h < mh)		// h'' < h'
            					{
						v_dash = edge[i].v;		// v' <- v
                				mh = ver[edge[i].v].h;		// h' <- h''
            					}

	            				// Push is only possible if height of adjacent
        	    				// is smaller than height of overflowing vertex
        	    				if (ver[u].h > ver[edge[i].v].h)			// if( h(u) > h' )	then do push operation
        	    				{

        	        			// Flow to be pushed is equal to minimum of
        	        			// remaining flow on edge and excess flow.
        	        			int flow = min(edge[i].capacity - edge[i].flow, e_dash);	// d <- min( c_f(u,v') , e' )
  	
	  	              			// Reduce excess flow for overflowing vertex
						#pragma omp atomic
                					ver[u].e_flow = ver[u].e_flow - flow;	    		// AtomicSub(c_f(u,v'), d)		
	  
	                			// Increase excess flow for adjacent
						#pragma omp atomic
                					ver[edge[i].v].e_flow = ver[edge[i].v].e_flow + flow;	// AtomicAdd(c_f(v',u), d)
	  
	                			// Add residual flow (With capacity 0 and negative
	                			// flow)
	                			#pragma omp atomic
                					edge[i].flow = edge[i].flow + flow;			// AtomicAdd(e(v'), d)	
	  
						#pragma omp critical
	                			updateReverseEdgeFlow(i, flow);					// AtomicSub(e(u), d)
	  
	                			flag = 1;
						break;
	            				}
		        		}
    		 		}

				if(!flag)						// do relabel operation
            	     		{
					// updating height of u
                			ver[u].h = mh + 1;			// h(u) <- h' + 1
				}

			   }
	
				// check for vertex with excess in the assigned chunk to a thread
		   		hasexcess = -1;
    				for (int i = startVx; i < endVx ; i++){
					if ((i == s) || (i == t)) continue;
					if (ver[i].e_flow > 0){
					hasexcess = i;
					break;
					}
				}
			
		      cycles--; 					// cycle <- cycle − 1
		      // cycle ends here...
		      }
			// ****Push_Relabel_Kernel Ends ****

    			#pragma omp barrier
    		} 		// parallel region ends here   
				
		totalitr = totalitr + 1;		
    }
 
    // ver.back() returns last Vertex, whose
    // e_flow will be final maximum flow
    return ver.back().e_flow;
}
  
// Serial BFS routinue
void BFS(vector < vector <int> > adjmat, int start, vector <int> &foreground) {
		int V = adjmat.size();
            vector <bool> visited(V,false);
            //vector <int> rv(V,-1);
            queue <int> q;
            
            //rv[start] = 0;
            visited[start] = true;
            q.push(start);
	    //foreground.push_back(start);         

            //list<int>::iterator i;            

            while(!q.empty()){
                int x = q.front();
                q.pop();
		foreground.push_back(x);
		// print order in which vertices are visited during the traversal.
		//cout << x << " ";
                for(int i=0;i<adjmat.size();i++){
                    if( !visited[i] && adjmat[x][i] != 0 ){
                     visited[i] = true;
                     //rv[*i] = rv[x] + 1;
                     q.push(i);    
                    }
                }
            }

        }

// Main function
int main(int argc, char** argv)
{
	    
		if (argc != 9)
		{
      		printf("\n A command line argument not proper\n Run code as follows: \n./a.out image_name bseed1 bseed2 bseed3 bseed4 bseed5 fseed1 num_threads \n For more details see README file \n ...exiting the program..\n\n");
      		return 1;
    		}

	    string image_name = argv[1];
	    int seed11 = stoi(argv[2]);
	    int seed12 = stoi(argv[3]);
	    int seed13 = stoi(argv[4]);
	    int seed14 = stoi(argv[5]);
	    int seed15 = stoi(argv[6]);

	    int seed2 = stoi(argv[7]);
	    struct timeval t1, t2;

	    int thread_count = 1;
	    thread_count = strtol(argv[8], NULL, 10);
    		

		// opening the image
	    Mat img = imread(image_name,IMREAD_GRAYSCALE);

		// getting the x and y dimensions of the image
	    int dim = img.rows;		 
		// delaring 2-D matrix to store the pixels in the image
	    vector < vector <int> > image(dim,vector <int>(dim,0));  

	gettimeofday(&t1, 0);

		// storing the pixels in 2-D matrix from the input image
	#pragma omp parallel num_threads(thread_count)
    	{
	    #pragma omp for schedule(static)
	    for (int i=0;i<img.rows;i++)
	    {
	      for(int j=0;j<img.cols;j++) 
      		{
        	image[i][j] = (int)img.at<uchar>(i,j);
      		}
		//cout << "\n";
    	    }			
	}	
	/*
	 for(int i=0;i<dim;i++){
		for(int j=0;j<dim;j++){
			cout << image[i][j] << " ";
		}
	cout << "\n";
	}	*/

	 int V = (dim*dim) + 2;
    	 Graph g(V);
	
   // create the adj-mat
   	vector < vector <int> > adjmat(V,vector <int>(V,0));

	// Compute the edge weights based on the pixel intensities..
	int max = 0;

	// Add the **n links**
	// add the edges in X directions
	int w;
    
	for(int i=0;i<(dim);i++)
	{
		for(int j=0;j<(dim-1);j++){
			w = (int)( 100.0*( 1 / exp( ( ( (double)image[i][j] - (double)image[i][j+1])*( (double)image[i][j] - (double)image[i][j+1]) ) / ( (double)2*30*30) ) ) );  	//B(p,q)
			if(w>max)
				max = w; 
			//cout << (i*dim+j)+1 << " " << (i*dim+j)+2  << " " << w << "\n";
			g.addEdge( (i*dim+j)+1 , (i*dim+j)+2 , w);
			g.addEdge( (i*dim+j)+2, (i*dim+j)+1  , w);		
		}
	}
    
	
	// add the edges in Y directionss
	for(int j=0;j<(dim);j++)
	{
		for(int i=0;i<(dim-1);i++){
			w = (int)( 100.0*( 1 / exp( ( ( (double)image[i][j] - (double)image[i+1][j])*( (double)image[i][j] - (double)image[i+1][j]) ) / ( (double)2*30*30) ) ) );  	//B(p,q)
			if(w>max)
				max = w;			
			//cout << (i*dim+j)+1 << " " << (i*dim+j)+dim+1  << " " << w << "\n";
			g.addEdge( (i*dim+j)+1 , (i*dim+j)+dim+1 , w);
			g.addEdge( (i*dim+j)+dim+1, (i*dim+j)+1  , w);
		}
	}

	// Add the **t links** using the seed1 and seed2
	g.addEdge( 0 , seed11 , max*100);					
	g.addEdge( seed11 , 0 , max*100);

	g.addEdge( 0 , seed12 , max*100);					
	g.addEdge( seed12 , 0 , max*100);

	g.addEdge( 0 , seed13 , max*100);					
	g.addEdge( seed13 , 0 , max*100);

	g.addEdge( 0 , seed14 , max*100);					
	g.addEdge( seed14 , 0 , max*100);

	g.addEdge( 0 , seed15 , max*100);					
	g.addEdge( seed15 , 0 , max*100);

	g.addEdge( seed2 , V-1 , max*100);
	g.addEdge( V-1, seed2 , max*100);
	
    // Initialize source and sink
    int s = 0, t = V-1;

    int n = g.edge.size();

   // add the edge to the adjmat
	for(int i=0;i<n;i++){
		adjmat[g.edge[i].u][g.edge[i].v] = g.edge[i].capacity;
	}	
	
	
	int mf =  g.getMaxFlow(s, t, thread_count);
	
		
    	// Print the total number of iteration
    	cout << "Total number of iterations need for convergence:" <<totalitr << "\n";
	
	// Print the Max flow value
	cout << "Maximum flow is " << mf << "\n";

	/*
	for(int i=0;i<n;i++){
		cout << g.edge[i].u << " " << g.edge[i].v << " " << g.edge[i].flow << "\n"; 
	}*/
	
	for(int i=0;i<n;i++){
		if(adjmat[g.edge[i].u][g.edge[i].v] != 0)
			adjmat[g.edge[i].u][g.edge[i].v] -= g.edge[i].flow;
	}
	/*
	for(int i=0;i<V;i++){
		for(int j=0;j<V;j++){
			cout << adjmat[i][j] << " ";					
		}	
	cout << "\n";
	}*/
	

	// Call BFS
	vector<int> foreground;
	BFS(adjmat,s,foreground);

	
	// print the foreground pixels
	int z1 = foreground.size();
	/*	
	cout << "\n";
	cout << "Following are the foregound pixels:\n";
	for(int i=0;i<z1;i++){
		cout << foreground[i] << " ";
	}

	cout << "\n";
	*/	

	// compute and print the background pixels
	set<int, greater<int> > s1;
	for(int i=0;i<V;i++){
		s1.insert(i);
	}
	for(int i=0;i<z1;i++){
		s1.erase(foreground[i]);
	}
	/*
	cout << "Following are the background pixels:\n";
	
	set<int, greater<int> >::iterator itr;
	for (itr = s1.begin(); itr != s1.end(); itr++)
	{
           cout << *itr<<" ";
	}
	cout << endl;	
	*/

	vector< char > finalimage(V-2,'B');
	for(int i=0;i<z1;i++){
		if( (foreground[i] == 0) || (foreground[i] == V-1) )
			continue;
		finalimage[ foreground[i]-1 ] = 'F';
	}

	int size = sqrt(V-2);
	/*
	for(int i=0;i<size;i++){
		for(int j=0;j<size;j++){
			cout << finalimage[ (i*size) + j ] << " ";
		}
	cout << "\n";
	}	*/

	 //Mat segmented_image(dim,dim, CV_8UC1, Scalar(0,0,0) );
	Mat segmented_image = Mat::zeros(dim,dim,CV_8UC3);

     #pragma omp parallel num_threads(thread_count)
     {
	#pragma omp for schedule(static)	
    	for (int i = 0; i < size; i++)
    	{
        	for (int j = 0; j < size; j++)
        	{
			Vec3b color = segmented_image.at<Vec3b>(Point(j,i));

			if( finalimage[ (i*size) + j ] == 'F' ){
		            	color.val[0] = 0;
        		    	color.val[1] = 255;
        		    	color.val[2] = 255;
			}			
			else{
				color.val[0] = 205;		// B
        		    	color.val[1] = 0;		// G
        		    	color.val[2] = 0;		// R
			}
			
			segmented_image.at<Vec3b>(Point(j,i)) = color;

         	}
    	}
     }

	gettimeofday(&t2, 0);
	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0; // Time taken 
        printf("\nTime taken by the Maximum flow algorithm to execute is: %.6f ms\n\n", time);
	
    	imwrite("output.tif",segmented_image);

	waitKey(0);
	
    return 0;
}
