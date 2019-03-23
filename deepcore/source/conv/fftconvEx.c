#include"../../include/conv/fftconv.h"

__local_func int idc_cellconv_choose_optimal_size( int nx, int ny, int fx, int fy )
{
    int iOpt, n=INT_MAX;
    for( int i=2; i>=0; i-- ){
        int cell_size=1<<(3+i);
        int dx=cell_size-fx+1;
        int dy=cell_size-fy+1;
        int grid_x=(nx+dx-1)/dx;
        int grid_y=(ny+dy-1)/dy;
        int size=grid_x*grid_y*cell_size*((cell_size>>1)+1);
        if(size<n){ iOpt=i; n=size; }
    }
    return iOpt;
}
__local_func int idc_fftconv_choose_optimal_size( int pnx, int pny, int qnx, int qny, int fx, int fy )
{
    int iOpt, a;
    int n=idc_minlds(pnx>pny?pnx:pny);
    if((n>128)||(n<=32)){
        return idc_cellconv_choose_optimal_size( qnx, qny, fx, fy );
    }
    a=n*((n>>1)+1);
    iOpt=n==64?3:4;
    for( int i=2; i>=0; i-- ){
        int cell_size=1<<(3+i);
        int dx=cell_size-fx+1;
        int dy=cell_size-fy+1;
        int grid_x=(qnx+dx-1)/dx;
        int grid_y=(qny+dy-1)/dy;
        int size=grid_x*grid_y*cell_size*((cell_size>>1)+1);
        if(size<a){ iOpt=i; a=size; }
    }
    return iOpt;
}