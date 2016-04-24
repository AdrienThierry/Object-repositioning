// Copyright 2001-2006 Vladimir Kolmogorov (v.kolmogorov@cs.ucl.ac.uk), Yuri Boykov (yuri@csd.uwo.ca).
// See README.txt

#include "graph.h"

#ifdef _MSC_VER
#pragma warning(disable: 4661)
#endif

// Instantiations: <captype, tcaptype, flowtype>
// IMPORTANT: 
//    flowtype should be 'larger' than tcaptype 
//    tcaptype should be 'larger' than captype

template class Graph<int,int,int>;
template class Graph<short,int,int>;
template class Graph<float,float,float>;
template class Graph<double,double,double>;

