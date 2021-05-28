// Program to implement image segmentation using push-relabel algorithm using graph cuts
/**
Dependencies:
OpenCV, C++, OpenMP

Constraints:
Input image should be a square grey-scale image

compile and Run:
g++ Image-Seg-Serial.cpp `pkg-config --cflags --libs opencv`
./a.out image_name bseed1 bseed2 bseed3 bseed4 bseed5 fseed1

*/
#include <bits/stdc++.h>
#include <math.h>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <omp.h>
#include <algorithm>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <string>
using namespace cv;
using namespace std;

int totalitr=0;
  
// structure to store the edge
struct Edge
{
    int flow, capacity;
    int u, v;
  
    Edge(int flow, int capacity, int u, int v)
    {
        this->flow = flow;
        this->capacity = capacity;
        this->u = u;
        this->v = v;
    }
};
  
// structure to store the vertices
struct Vertex
{
    int h, e_flow;
  
    Vertex(int h, int e_flow)
    {
        this->h = h;
        this->e_flow = e_flow;
    }
};
  
// class representing flow network
class Graph
{
public:
    int V;   
    vector<Vertex> ver;
    vector<Edge> edge;
  
    void updateReverseEdgeFlow(int i, int flow);
  
    Graph(int V); 

    void addEdge(int u, int v, int w);
    int getMaxFlow(int s, int t);
};
  
Graph::Graph(int V)
{
    this->V = V;
  
    // all vertices are initialized with 0 height
    // and 0 excess flow
    for (int i = 0; i < V; i++)
        ver.push_back(Vertex(0, 0));
}
  
// Initialize the flow at all the edges to 0.
void Graph::addEdge(int u, int v, int capacity)
{
    edge.push_back(Edge(0, capacity, u, v));
}
  
// function to search for overflowing vertices
int overFlowVertex(vector<Vertex>& ver)
{
    for (int i = 1; i < ver.size() - 1; i++)
       if (ver[i].e_flow > 0)
            return i;
  
    // No overflowing vertex present then..
    return -1;
}

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

    Edge e = Edge(0, flow, u, v);
    edge.push_back(e);
}
  
// Serial function for finding max-flow, min-cut of graph
int Graph::getMaxFlow(int s, int t)
{
    ver[s].h = ver.size();		// Initialize the Heights

    // create a pre-flow \textit{f} that saturates all out-going edges of s 
    for (int i = 0; i < edge.size(); i++)		// \forall (s, u) \in E do
    {
        if (edge[i].u == s)
        {
            edge[i].flow = edge[i].capacity;		// c_f (s, u) <- c_f (s, u) - c_{su}
            ver[edge[i].v].e_flow += edge[i].flow;	// e(u) <- c_{su}
 
            edge.push_back(Edge(-edge[i].flow, 0, edge[i].v, s));		// c_f (u, s) <- c_f (u, s) + c_{su}
        }
    }
   
    // while there exists vertex with excess flow on it
    while (overFlowVertex(ver) != -1)			// \While{$\exists$ v $\neq$ s, t with e(v) $>$ 0 }
    {
        int u = overFlowVertex(ver);
        // Initialize minimum height of an adjacent vertex
    	int mh = INT_MAX;				// h' <- INF
	int e_dash = ver[u].e_flow;			// e' <- e(u)
	int v_dash;

	int flag = 0;
	
  	for (int i = 0; i < edge.size(); i++)			// \forall (u, v) \in E
    	{
        	if (edge[i].u == u)
        	{
            			if (edge[i].flow == edge[i].capacity)
                			continue;

				// update minimum height 
            			if (ver[edge[i].v].h < mh)		// h'' < h'
            			{
				v_dash = edge[i].v;		// v' <- v
                		mh = ver[edge[i].v].h;		// h' <- h''
            			}

				// if there exist any neighouring vertex with lower height then perform push operation
				// \eIf{$\exists$ (v, w) $\in$ E with h(v) = h(w) + 1 } then do push operation
        	    		if (ver[u].h > ver[edge[i].v].h)			// if( h(u) > h' )	then do push operation
        	    		{

        	        	int flow = min(edge[i].capacity - edge[i].flow, e_dash);	// d <- min( c_f(u,v') , e' )
  	
                		ver[u].e_flow = ver[u].e_flow - flow;	    		// Sub(c_f(u,v'), d)		
	  
                		ver[edge[i].v].e_flow = ver[edge[i].v].e_flow + flow;	// Add(c_f(v',u), d)
	             
                		edge[i].flow = edge[i].flow + flow;			// Add(e(v'), d)	
	  
	               		updateReverseEdgeFlow(i, flow);					// Sub(e(u), d)
	  
	        		flag = 1;
				break;
	            		}
		}
    	}

	if(!flag)						// else do relabel operation
        {
	// updating height of u
        ver[u].h = mh + 1;			// h(u) <- h' + 1
	}

	totalitr += 1;
    }	
  
    // excess flow on the sink vertex is max-flow
    return ver.back().e_flow;
}
  
// BFS routinue
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
	    
		if (argc != 8)
    		{
      		printf("\n A command line argument not proper\n Run code as follows: \n./a.out image_name bseed1 bseed2 bseed3 bseed4 bseed5 fseed1 \n For more details see README file \n ...exiting the program..\n\n");
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

		// opening the image
	    Mat img = imread(image_name,IMREAD_GRAYSCALE);

		// getting the x and y dimensions of the image
	    int dim = img.rows;		 
		// delaring 2-D matrix to store the pixels in the image
	    vector < vector <int> > image(dim,vector <int>(dim,0));  

	gettimeofday(&t1, 0);

		// storing the pixels in 2-D matrix from the input image

	// ****** Image Reading Begins here ******

	    for (int i=0;i<img.rows;i++)
	    {
	      for(int j=0;j<img.cols;j++) 
      		{
        	image[i][j] = (int)img.at<uchar>(i,j);
      		}
		//cout << "\n";
    	    }			
		
	/*
	 for(int i=0;i<dim;i++){
		for(int j=0;j<dim;j++){
			cout << image[i][j] << " ";
		}
	cout << "\n";
	}	*/

	// ****** Image Reading Ends here ******

	// ****** Converting Image to Graph begins ******

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

	// ****** Converting Image to Graph ends ******

	// ******** Call Serial Push-Relabel Algorithm ********	
	int mf =  g.getMaxFlow(s, t);
		
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
	

	// ********* Call BFS *********
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

	// ******** Displaying the Segmented Image begins **********

	 //Mat segmented_image(dim,dim, CV_8UC1, Scalar(0,0,0) );
	Mat segmented_image = Mat::zeros(dim,dim,CV_8UC3);
	
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
     
	// ******** Displaying the Segmented Image ends **********

	gettimeofday(&t2, 0);
	double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0; // Time taken
        printf("\nTime taken by the Maximum flow algorithm to execute is: %.6f ms\n\n", time);
	
    	imwrite("output.tif",segmented_image);

	waitKey(0);
	
    return 0;
}
