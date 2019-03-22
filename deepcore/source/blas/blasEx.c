#include"../../include/blas/blasEx.h"

__local_func unsigned int idc_get_optimal_sgemm_id( int nx, int ny, int n )
{
    int dx[3], dy, axis_x, axis_y;
    dx[0]=(nx+ 31)>>5;
    dx[1]=(nx+ 63)>>6;
    dx[2]=(nx+127)>>7;
    axis_y=(ny>32)+(ny>64);
    dy=(ny+(1<<(5+axis_y))-1)>>(5+axis_y);
    for( axis_x=2; axis_x>=0; --axis_x ){
        if((dy*dx[axis_x]>=n)||(axis_x==0)) break;
    }
    if((axis_x<2)&&(axis_y==2)){
        if(((dx[axis_x]*((ny+255)>>8))>=n)&&(ny>128)){ axis_y=3; }
    }
    return ((axis_y<<4)|axis_x);
}
__local_func unsigned int idc_get_optimal_cgemm_id( int nx, int ny, int n, int bat )
{
    int dx[4], dy[4], axis_x, axis_y;
    dx[0]=(nx+ 15)>>4;
    dx[1]=(nx+ 31)>>5;
    dx[2]=(nx+ 63)>>6;
    dx[3]=(nx+127)>>7;
    dy[0]=(ny+ 31)>>5;
    dy[1]=(ny+ 63)>>6;
    dy[2]=(ny+127)>>7;
    dy[3]=(ny+255)>>8;
    if(ny<=32){
        axis_y=0;
        for( axis_x=(nx>16)+(nx>32)+(nx>64); axis_x>=0; --axis_x ){ if((bat*dx[axis_x]>=n)||(axis_x==0)) break; }
    } else {
        if((nx<=16)||((nx>32)&&(nx<=48))){
            axis_x=0;
            for( axis_y=(ny>32)+(ny>64)+(ny>128); axis_y>=0; --axis_y ){ if((bat*dx[axis_x]*dy[axis_y]>=n)||(axis_y==0)) break; }
        } else
        if(((nx>16)&&(nx<=32))||((nx>64)&&(nx<=96))){ 
            axis_x=1;
            for( axis_y=(ny>32)+(ny>64); axis_y>=0; --axis_y ){ if((bat*dx[axis_x]*dy[axis_y]>=n)||(axis_y==0)) break; }
        } else {
            axis_x=2; axis_y=1;
        }
    }
    return ((axis_y<<4)|axis_x);
}