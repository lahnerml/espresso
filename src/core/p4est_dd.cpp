#include "p4est_dd.hpp"
//--------------------------------------------------------------------------------------------------
#ifdef USE_P4EST
//--------------------------------------------------------------------------------------------------
#ifdef LEES_EDWARDS
#error "p4est and Lees-Edwards are not compatible yet."
#endif
//--------------------------------------------------------------------------------------------------
#include <p4est_to_p8est.h>
#include <p8est_ghost.h>
#include <p8est_mesh.h>
#include <p8est_vtk.h>
#include <p8est_extended.h>
#include <vector>
#include "call_trace.hpp"
#include "domain_decomposition.hpp"
#include "ghosts.hpp"
//--------------------------------------------------------------------------------------------------
#define CELLS_MAX_NEIGHBORS 14
//--------------------------------------------------------------------------------------------------
typedef struct {
  int64_t idx;
  int64_t rank;
  int shell;
  int boundary;
  int neighbor[26];
  int coord[3];
} local_shell_t;
//--------------------------------------------------------------------------------------------------
typedef struct {
  int rank;
  int cnt;
  uint64_t dir;
  int *idx;
} comm_t;
//--------------------------------------------------------------------------------------------------
static p4est_t *p4est = NULL;
static p4est_ghost_t *p4est_ghost = NULL;
static p4est_mesh_t *p4est_mesh = NULL;
static p4est_connectivity_t *p4est_conn  = NULL;
static local_shell_t *p4est_shell = NULL;
//--------------------------------------------------------------------------------------------------
static int brick_size[3];
//--------------------------------------------------------------------------------------------------
static comm_t *comm_send = NULL;
static comm_t *comm_recv = NULL;
static int num_comm_send = 0;
static int num_comm_recv = 0;
static int *comm_proc = NULL;
static int *comm_rank = NULL;
static int num_comm_proc = 0;
//--------------------------------------------------------------------------------------------------
static size_t num_cells = 0;
static size_t num_local_cells = 0;
static size_t num_ghost_cells = 0;
//--------------------------------------------------------------------------------------------------
static const int neighbor_lut[3][3][3] = {
  {{18, 6,19},{10, 4,11},{20, 7,21}},
  {{14, 2,15},{ 0,-1, 1},{16, 3,17}},
  {{22, 8,23},{12, 5,13},{24, 9,25}}
};
//--------------------------------------------------------------------------------------------------
static const int half_neighbor_idx[14] = {
  -1, 1, 16, 3, 17, 22, 8, 23, 12, 5, 13, 24, 9, 25
};
//--------------------------------------------------------------------------------------------------
typedef struct {
  int rank;
  int lidx;
  int ishell[26];
  int rshell[26];
} quad_data_t;
//--------------------------------------------------------------------------------------------------
static void init_fn (p4est_t* p4est, p4est_topidx_t tree, p4est_quadrant_t* q) {
  ((quad_data_t*)(q->p.user_data))->rank = this_node;
  ((quad_data_t*)(q->p.user_data))->lidx = -1;
  for (int i=0;i<26;++i) {
    ((quad_data_t*)(q->p.user_data))->ishell[i] = -1;
    ((quad_data_t*)(q->p.user_data))->rshell[i] = -1;
  }
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_free () {
  CALL_TRACE();
  
  if (p4est_mesh)
    p4est_mesh_destroy(p4est_mesh);
  if (p4est_ghost)
    p4est_ghost_destroy(p4est_ghost);
  if (p4est)
    p4est_destroy (p4est);
  if (p4est_conn)
    p4est_connectivity_destroy(p4est_conn);
  if (p4est_shell)
    delete[] p4est_shell;
  p4est = NULL;
  p4est_ghost = NULL;
  p4est_mesh = NULL;
  p4est_conn = NULL;
  p4est_shell = NULL;
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_create_grid () {
  CALL_TRACE();
  
  quad_data_t *ghost_data;  
  
  dd_p4est_free();
  
  int t_x, t_y, t_z, grid_level;
  grid_level = 0;
  
  // compute number of cells
  t_x = t_y = t_z = 1;
  if (max_range > 0) {
    if (box_l[0] > max_range) t_x = box_l[0]/max_range;
    if (box_l[1] > max_range) t_y = box_l[1]/max_range;
    if (box_l[2] > max_range) t_z = box_l[2]/max_range;
    if (t_x < 1) t_x = 1;
    if (t_y < 1) t_y = 1;
    if (t_z < 1) t_z = 1;
  }
  
#ifndef P4EST_NOCHANGE
  // set global variables
  dd.cell_size[0] = box_l[0]/t_x;
  dd.cell_size[1] = box_l[1]/t_y;
  dd.cell_size[2] = box_l[2]/t_z;
  dd.inv_cell_size[0] = 1.0/dd.cell_size[0];
  dd.inv_cell_size[1] = 1.0/dd.cell_size[1];
  dd.inv_cell_size[2] = 1.0/dd.cell_size[2];
  max_skin = std::min(std::min(dd.cell_size[0],dd.cell_size[1]),dd.cell_size[2]) - max_cut;
#endif
  
  /*t_x = dd.cell_grid[0]*node_grid[0];
  t_y = dd.cell_grid[1]*node_grid[1];
  t_z = dd.cell_grid[2]*node_grid[2];*/
  
  brick_size[0] = t_x;
  brick_size[1] = t_y;
  brick_size[2] = t_z;
  
  // divide all dimensions by biggest common power of 2
  while (((t_x|t_y|t_z)&1) == 0) {
    ++grid_level;
    t_x >>= 1;
    t_y >>= 1;
    t_z >>= 1;
  }
  
  // create p4est structs
  p4est_conn = p8est_connectivity_new_brick (t_x, t_y, t_z, 
                                             PERIODIC(0), PERIODIC(1), PERIODIC(2));
  p4est = p4est_new_ext (sc_MPI_COMM_WORLD, p4est_conn, 0, grid_level, true, 
                         sizeof(quad_data_t), init_fn, NULL);
  p4est_partition(p4est, 0, NULL);
  p4est_ghost = p4est_ghost_new(p4est, P8EST_CONNECT_CORNER);
  p4est_mesh = p4est_mesh_new_ext(p4est, p4est_ghost, 1, 1, 0, P8EST_CONNECT_CORNER);
  
  // write cells to VTK
  p4est_vtk_write_file (p4est, NULL, P4EST_STRING "_lj");
  
  printf("%i : %i %i-%i %i\n",
    this_node,periodic,p4est->first_local_tree,p4est->last_local_tree,p4est->local_num_quadrants);
  
  // geather cell neighbors
  std::vector<uint64_t> quads;
  std::vector<local_shell_t> shell;
  quads.clear();
  shell.clear();
  
  /*for (int i=0;i<p4est->local_num_quadrants;++i) {
    p4est_quadrant_t *q = p4est_mesh_get_quadrant(p4est,p4est_mesh,i);
    data = (quad_data_t*)(q->p.user_data);
    double xyz[3];
    p4est_qcoord_to_vertex(p4est_conn, p4est_mesh->quad_to_tree[i], q->x, q->y, q->z, xyz);
    uint64_t ql = 1<<p4est_tree_array_index(p4est->trees,p4est_mesh->quad_to_tree[i])->maxlevel;
    uint64_t x = xyz[0]*ql;
    uint64_t y = xyz[1]*ql;
    uint64_t z = xyz[2]*ql;
    quads.push_back((x+1) | ((y+1)<<21) | ((z+1)<<42));
    local_shell_t ls;
    ls.idx = i;
    ls.rank = this_node;
    ls.shell = 0;
    shell.push_back(ls);
    for (int n=0;n<26;++n) {
      sc_array_t ne, ni;
      sc_array_init(&ne,sizeof(int));
      sc_array_init(&ni,sizeof(int));
      p4est_mesh_get_neighbors(p4est, p4est_ghost, p4est_mesh, i, -1, n, 0, NULL, &ne, &ni);
      if (ni.elem_count > 1)
        printf("%i %i %li strange stuff\n",i,n,ni.elem_count);
      if (ni.elem_count > 0) {
        data->ishell[n] = ni.array[0];
        if (ne.array[n] >= 0)
          data->rshell[n] = this_node;
        else {
          data->rshell[n] = p4est_mesh->ghost_to_proc[ni.array[0]];
          printf("%i %i remote %i\n",i,n,ne.array[n]);
        }
      }
    }
  }*/
  
  for (int i=0;i<p4est->local_num_quadrants;++i) {
    p4est_mesh_face_neighbor_t mfn;
    int tidx = -1;
    int qidx, ridx, fidx;
    p4est_mesh_quadrant_cumulative (p4est, p4est_mesh, i, &tidx, &qidx);
    p4est_mesh_face_neighbor_init2(&mfn, p4est, p4est_ghost, p4est_mesh, tidx, qidx);
    int fcnt = 0;
    p4est_quadrant_t *q = p4est_mesh_get_quadrant(p4est,p4est_mesh,i);
    quad_data_t *data = (quad_data_t*)(q->p.user_data);
    data->lidx = i;
    data->rank = this_node;
    double xyz[3];
    p4est_qcoord_to_vertex(p4est_conn, p4est_mesh->quad_to_tree[i], q->x, q->y, q->z, xyz);
    uint64_t ql = 1<<p4est_tree_array_index(p4est->trees,p4est_mesh->quad_to_tree[i])->maxlevel;
    uint64_t x = xyz[0]*ql;
    uint64_t y = xyz[1]*ql;
    uint64_t z = xyz[2]*ql;
    quads.push_back((x+1) | ((y+1)<<21) | ((z+1)<<42));
    local_shell_t ls;
    ls.idx = i;
    ls.rank = this_node;
    ls.shell = 0;
    ls.boundary = 0;
    ls.coord[0] = x;
    ls.coord[1] = y;
    ls.coord[2] = z;
    for (int n=0;n<26;++n) ls.neighbor[n] = -1;
    /*if (PERIODIC(0) && x == 0) ls.boundary |= 1;
    if (PERIODIC(0) && x == brick_size[0] - 1) ls.boundary |= 2;
    if (PERIODIC(1) && y == 0) ls.boundary |= 4;
    if (PERIODIC(1) && y == brick_size[1] - 1) ls.boundary |= 8;
    if (PERIODIC(2) && z == 0) ls.boundary |= 16;
    if (PERIODIC(2) && z == brick_size[2] - 1) ls.boundary |= 32;*/
    shell.push_back(ls);
    while (p4est_mesh_face_neighbor_next(&mfn, &tidx, &qidx,&fidx,&ridx) != NULL) {
      if (mfn.current_qtq == i) continue;
      fcnt = mfn.face-1;
      if (ridx != this_node)
        data->ishell[fcnt] = p4est_quadrant_array_index(&p4est_ghost->ghosts,qidx)->p.piggy3.local_num;//qidx - p4est_ghost->proc_offsets[ridx];
      else
        data->ishell[fcnt] = p4est_tree_array_index(p4est->trees,tidx)->quadrants_offset + qidx;
      data->rshell[fcnt] = ridx;
    }
  }
  
  ghost_data = P4EST_ALLOC (quad_data_t, p4est_ghost->ghosts.elem_count);
  p4est_ghost_exchange_data (p4est, p4est_ghost, ghost_data);
  for (int i=0;i<p4est->local_num_quadrants;++i) {
    p4est_mesh_face_neighbor_t mfn;
    int tidx = -1;
    int qidx, ridx, fidx;
    p4est_mesh_quadrant_cumulative (p4est, p4est_mesh, i, &tidx, &qidx);
    p4est_mesh_face_neighbor_init2(&mfn, p4est, p4est_ghost, p4est_mesh, tidx, qidx);
    int fcnt = 0;
    quad_data_t *data = (quad_data_t*)p4est_mesh_get_quadrant(p4est,p4est_mesh,i)->p.user_data;
    while (p4est_mesh_face_neighbor_next(&mfn, &tidx, &qidx,&fidx,&ridx) != NULL) {
      if (mfn.current_qtq == i) continue;
      fcnt = mfn.face-1;
      quad_data_t* ndata = (quad_data_t*)p4est_mesh_face_neighbor_data(&mfn,ghost_data);
      switch(fcnt) {
      case 0:
        data->ishell[6 + 8] = ndata->ishell[2];
        data->rshell[6 + 8] = ndata->rshell[2];
        data->ishell[6 + 4] = ndata->ishell[4];
        data->rshell[6 + 4] = ndata->rshell[4];
        break;
      case 1:
        data->ishell[6 + 11] = ndata->ishell[3];
        data->rshell[6 + 11] = ndata->rshell[3];
        data->ishell[6 + 7] = ndata->ishell[5];
        data->rshell[6 + 7] = ndata->rshell[5];
        break;
      case 2:
        data->ishell[6 + 9] = ndata->ishell[1];
        data->rshell[6 + 9] = ndata->rshell[1];
        data->ishell[6 + 0] = ndata->ishell[4];
        data->rshell[6 + 0] = ndata->rshell[4];
        break;
      case 3:
        data->ishell[6 + 10] = ndata->ishell[0];
        data->rshell[6 + 10] = ndata->rshell[0];
        data->ishell[6 + 3] = ndata->ishell[5];
        data->rshell[6 + 3] = ndata->rshell[5];
        break;
      case 4:
        data->ishell[6 + 1] = ndata->ishell[3];
        data->rshell[6 + 1] = ndata->rshell[3];
        data->ishell[6 + 5] = ndata->ishell[1];
        data->rshell[6 + 5] = ndata->rshell[1];
        break;
      case 5:
        data->ishell[6 + 6] = ndata->ishell[0];
        data->rshell[6 + 6] = ndata->rshell[0];
        data->ishell[6 + 2] = ndata->ishell[2];
        data->rshell[6 + 2] = ndata->rshell[2];
        break;
      };
    }
  }
  p4est_ghost_exchange_data (p4est, p4est_ghost, ghost_data);
  for (int i=0;i<p4est->local_num_quadrants;++i) {
    p4est_mesh_face_neighbor_t mfn;
    int tidx = -1;
    int qidx, ridx, fidx;
    p4est_mesh_quadrant_cumulative (p4est, p4est_mesh, i, &tidx, &qidx);
    p4est_mesh_face_neighbor_init2(&mfn, p4est, p4est_ghost, p4est_mesh, tidx, qidx);
    int fcnt = 0;
    quad_data_t* data = (quad_data_t*)p4est_mesh_get_quadrant(p4est,p4est_mesh,i)->p.user_data;
    while (p4est_mesh_face_neighbor_next(&mfn, &tidx, &qidx,&fidx,&ridx) != NULL) {
      if (mfn.current_qtq == i) continue;
      fcnt = mfn.face-1;
      quad_data_t* ndata = (quad_data_t*)p4est_mesh_face_neighbor_data(&mfn,ghost_data);
      switch(fcnt) {
      case 0:
        data->ishell[18 + 0] = ndata->ishell[6 + 0];
        data->rshell[18 + 0] = ndata->rshell[6 + 0];
        data->ishell[18 + 2] = ndata->ishell[6 + 1];
        data->rshell[18 + 2] = ndata->rshell[6 + 1];
        data->ishell[18 + 4] = ndata->ishell[6 + 2];
        data->rshell[18 + 4] = ndata->rshell[6 + 2];
        data->ishell[18 + 6] = ndata->ishell[6 + 3];
        data->rshell[18 + 6] = ndata->rshell[6 + 3];
        break;
      case 1:
        data->ishell[18 + 1] = ndata->ishell[6 + 0];
        data->rshell[18 + 1] = ndata->rshell[6 + 0];
        data->ishell[18 + 3] = ndata->ishell[6 + 1];
        data->rshell[18 + 3] = ndata->rshell[6 + 1];
        data->ishell[18 + 5] = ndata->ishell[6 + 2];
        data->rshell[18 + 5] = ndata->rshell[6 + 2];
        data->ishell[18 + 7] = ndata->ishell[6 + 3];
        data->rshell[18 + 7] = ndata->rshell[6 + 3];
        break;
      };
    }
  }
  P4EST_FREE (ghost_data);
    
  char fname[100];
  sprintf(fname,"cells_%i.list",this_node);
  FILE* h = fopen(fname,"w");
  
  // compute ghost, mirror and boundary information
  for (int i=0;i<p4est->local_num_quadrants;++i) {
    p4est_quadrant_t* q = p4est_mesh_get_quadrant(p4est,p4est_mesh,i);
    quad_data_t *data = (quad_data_t*)(q->p.user_data);
    double xyz[3];
    p4est_qcoord_to_vertex(p4est_conn, p4est_mesh->quad_to_tree[i], q->x, q->y, q->z, xyz);
    uint64_t ql = 1<<p4est_tree_array_index(p4est->trees,p4est_mesh->quad_to_tree[i])->maxlevel;
    uint64_t x = xyz[0]*ql;
    uint64_t y = xyz[1]*ql;
    uint64_t z = xyz[2]*ql;
    fprintf(h,"%i %lix%lix%li ",i,x,y,z);
    for (uint64_t zi=0;zi <= 2;zi++)
      for (uint64_t yi=0;yi <= 2;yi++)
        for (uint64_t xi=0;xi <= 2;xi++) {
          if (xi == 1 && yi == 1 && zi == 1) continue;
          uint64_t qidx = (x+xi) | ((y+yi)<<21) | ((z+zi)<<42);
          size_t pos = 0;
          while (pos < quads.size() && quads[pos] != qidx) ++pos;
          if (pos == quads.size()) {
            quads.push_back(qidx);
            local_shell_t ls;
            ls.idx = data->ishell[neighbor_lut[zi][yi][xi]];
            ls.rank = data->rshell[neighbor_lut[zi][yi][xi]];
            ls.shell = 2;
            ls.boundary = 0;
            for (int n=0;n<26;++n) ls.neighbor[n] = -1;
            if (PERIODIC(0) && (x + xi) == 0) ls.boundary |= 1;
            if (PERIODIC(0) && (x + xi) == brick_size[0] + 1) ls.boundary |= 2;
            if (PERIODIC(1) && (y + yi) == 0) ls.boundary |= 4;
            if (PERIODIC(1) && (y + yi) == brick_size[1] + 1) ls.boundary |= 8;
            if (PERIODIC(2) && (z + zi) == 0) ls.boundary |= 16;
            if (PERIODIC(2) && (z + zi) == brick_size[2] + 1) ls.boundary |= 32;
            if (xi == 0 && yi == 1 && zi == 1) shell[i].boundary |= 1;
            if (xi == 2 && yi == 1 && zi == 1) shell[i].boundary |= 2;
            if (xi == 1 && yi == 0 && zi == 1) shell[i].boundary |= 4;
            if (xi == 1 && yi == 2 && zi == 1) shell[i].boundary |= 8;
            if (xi == 1 && yi == 1 && zi == 0) shell[i].boundary |= 16;
            if (xi == 1 && yi == 1 && zi == 2) shell[i].boundary |= 32;
            /*if (shell[i].boundary != 0) {
              if (xi == 0) ls.boundary |= 1;
              if (xi == 2) ls.boundary |= 2;
              if (yi == 0) ls.boundary |= 4;
              if (yi == 2) ls.boundary |= 8;
              if (zi == 0) ls.boundary |= 16;
              if (zi == 2) ls.boundary |= 32;
            }*/
            //ls.boundary &= shell[i].boundary;
            shell[i].neighbor[neighbor_lut[zi][yi][xi]] = shell.size();
            ls.coord[0] = int(x + xi) - 1;
            ls.coord[1] = int(y + yi) - 1;
            ls.coord[2] = int(z + zi) - 1;
            shell.push_back(ls);
            shell[i].shell = 1;
          } else {
            if (shell[pos].shell == 2) {
              shell[i].shell = 1;
              if (xi == 0 && yi == 1 && zi == 1) shell[i].boundary |= 1;
              if (xi == 2 && yi == 1 && zi == 1) shell[i].boundary |= 2;
              if (xi == 1 && yi == 0 && zi == 1) shell[i].boundary |= 4;
              if (xi == 1 && yi == 2 && zi == 1) shell[i].boundary |= 8;
              if (xi == 1 && yi == 1 && zi == 0) shell[i].boundary |= 16;
              if (xi == 1 && yi == 1 && zi == 2) shell[i].boundary |= 32;
              /*int tmp = 0;
              if (shell[i].boundary != 0) {
                if (xi == 0) tmp |= 1;
                if (xi == 2) tmp |= 2;
                if (yi == 0) tmp |= 4;
                if (yi == 2) tmp |= 8;
                if (zi == 0) tmp |= 16;
                if (zi == 2) tmp |= 32;
              }
              tmp &= shell[i].boundary;
              if (shell[pos].boundary > tmp)
                shell[pos].boundary = tmp;*/
            }
            shell[i].neighbor[neighbor_lut[zi][yi][xi]] = pos;
          }
        }
    for (int n=0;n<26;++n) {
      if (n < 6)
        fprintf(h,"f%i[",n);
      else if (n < 18)
        fprintf(h,"e%i[",n-6);
      else
        fprintf(h,"c%i[",n-18);
      if (data->ishell[n] >= 0)
        fprintf(h,"%i:%i] ",data->rshell[n],data->ishell[n]);
      else
        fprintf(h,"] ");
    }
    fprintf(h,"\n");
  }
  
  fclose(h);
  
  num_cells = (size_t)quads.size();
  num_local_cells = (size_t)p4est->local_num_quadrants;
  num_ghost_cells = num_cells - (size_t)p4est->local_num_quadrants;
  
  p4est_shell = new local_shell_t[num_cells];
  for (int i=0;i<num_cells;++i)
    p4est_shell[i] = shell[i];
  
  printf("%d : %ld, %ld, %ld\n",this_node,num_cells,num_local_cells,num_ghost_cells);

#ifndef P4EST_NOCHANGE  
  // allocate memory
  realloc_cells(num_cells);
  realloc_cellplist(&local_cells, local_cells.n = num_local_cells);
  realloc_cellplist(&ghost_cells, ghost_cells.n = num_ghost_cells);
#endif

}
//--------------------------------------------------------------------------------------------------
void dd_p4est_comm () {
  CALL_TRACE();
  
  std::vector<int>      send_idx[n_nodes];
  std::vector<int>      recv_idx[n_nodes];
  std::vector<uint64_t> send_tag[n_nodes];
  std::vector<uint64_t> recv_tag[n_nodes];
  uint64_t send_cnt_tag[n_nodes];
  uint64_t recv_cnt_tag[n_nodes];
  int send_cnt[n_nodes][64];
  int recv_cnt[n_nodes][64];
  
  int num_send = 0;
  int num_recv = 0;
  uint32_t num_send_flag = 0;
  uint32_t num_recv_flag = 0;
  
  num_comm_proc = 0;
  if (comm_proc) delete[] comm_proc;
  comm_proc = new int[n_nodes];
  
  for (int i=0;i<n_nodes;++i) {
    comm_proc[i] = -1;
    send_idx[i].clear();
    recv_idx[i].clear();
    send_cnt_tag[i] = 0;
    recv_cnt_tag[i] = 0;
    for (int j=0;j<64;++j) {
      send_cnt[i][j] = 0;
      recv_cnt[i][j] = 0;
    }
  }
  // create send and receive list
  char fname[100];
  sprintf(fname,"cells_conn_%i.list",this_node);
  FILE* h = fopen(fname,"w");
  for (int i=0;i<num_cells;++i) {
    // is ghost cell -> add to recv list
    if (p4est_shell[i].rank >= 0 && p4est_shell[i].shell == 2) {
      int irank = p4est_shell[i].rank;
      int pos = 0;
      while (pos < recv_idx[irank].size() && 
        p4est_shell[recv_idx[irank][pos]].idx < p4est_shell[i].idx) pos++;
      if (pos >= recv_idx[irank].size()) {
        recv_idx[irank].push_back(i);
        recv_tag[irank].push_back(1L<<p4est_shell[i].boundary);
      } else if (p4est_shell[recv_idx[irank][pos]].idx != p4est_shell[i].idx) {
        recv_idx[irank].insert(recv_idx[irank].begin() + pos, i);
        recv_tag[irank].insert(recv_tag[irank].begin() + pos, 1L<<p4est_shell[i].boundary);
      } else {
        recv_tag[irank][pos] |= 1L<<p4est_shell[i].boundary;        
      }
      recv_cnt[irank][p4est_shell[i].boundary] += 1;
      if ((recv_cnt_tag[irank] & (1L<<p4est_shell[i].boundary)) == 0) {
        ++num_recv;
        recv_cnt_tag[irank] |= 1L<<p4est_shell[i].boundary;
      }
      //recv_cnt[p4est_shell[i].rank].insert(recv_cnt[p4est_shell[i].rank].begin() + pos, i); //p4est_shell[i].idx);
      if ((num_recv_flag & (1<<irank)) == 0) {
          //++num_recv;
          comm_proc[irank] = num_comm_proc;
          num_comm_proc += 1;
          num_recv_flag |= 1<<irank;
      }
    }
    // is mirror cell -> add to send list
    if (p4est_shell[i].shell == 1) {
      //for (int n=0;n<n_nodes;++n) comm_cnt[n] = 0;
      for (int n=0;n<26;++n) {
        int nidx = p4est_shell[i].neighbor[n];
        int nrank = p4est_shell[nidx].rank;
        if (nidx < 0 || nrank < 0) continue;
        if (p4est_shell[nidx].shell != 2) continue;
        if (!send_tag[nrank].empty() && send_idx[nrank].back() == i) {
          if ((send_tag[nrank].back() & (1L<<p4est_shell[nidx].boundary))) continue;
          send_tag[nrank].back() |= (1L<<p4est_shell[nidx].boundary);
        } else {
          send_idx[nrank].push_back(i);
          send_tag[nrank].push_back(1L<<p4est_shell[nidx].boundary);
        }
        send_cnt[nrank][p4est_shell[nidx].boundary] += 1;
        if ((send_cnt_tag[nrank] & (1L<<p4est_shell[nidx].boundary)) == 0) {
          ++num_send;
          send_cnt_tag[nrank] |= 1L<<p4est_shell[nidx].boundary;
        }
        if ((num_send_flag & (1<<nrank)) == 0) {
          //++num_send;
          num_send_flag |= 1<<nrank;
        }
      }
    }
    fprintf(h,"%i %li:%li %i %i [ ",i,p4est_shell[i].rank,p4est_shell[i].idx,
      p4est_shell[i].shell,p4est_shell[i].boundary);
    for (int n=0;n<26;++n) fprintf(h,"%i ",p4est_shell[i].neighbor[n]);
    fprintf(h,"]\n");
  }
  fclose(h);
  sprintf(fname,"send_%i.list",this_node);
  h = fopen(fname,"w");
  for (int n=0;n<n_nodes;++n)
    for (int i=0;i<send_idx[n].size();++i)
      fprintf(h,"%i:%i 0x%016lx\n",n,send_idx[n][i],send_tag[n][i]);
  fclose(h);
  sprintf(fname,"recv_%i.list",this_node);
  h = fopen(fname,"w");
  for (int n=0;n<n_nodes;++n)
    for (int i=0;i<recv_idx[n].size();++i)
      fprintf(h,"%i:%li 0x%016lx\n",n,p4est_shell[recv_idx[n][i]].idx,recv_tag[n][i]);
  fclose(h);
  
  // prepare communicator
  printf("%i : proc %i send %i, recv %i\n",this_node,num_comm_proc,num_send,num_recv);
  if (comm_send) {
    for (int i=0;i<num_comm_send;++i)
      delete[] comm_send[i].idx;
    delete[] comm_send;
    comm_send = NULL;
  }
  if (comm_recv) {
    for (int i=0;i<num_comm_recv;++i)
      delete[] comm_recv[i].idx;
    delete[] comm_recv;
    comm_recv = NULL;
  }
  if (comm_rank) delete[] comm_rank;
  num_comm_recv = num_recv;
  num_comm_send = num_send;
  comm_recv = new comm_t[num_recv];
  comm_send = new comm_t[num_send];
  comm_rank = new int[num_comm_proc];
  for (int n=0,s_cnt=0,r_cnt=0;n<n_nodes;++n) {
    if (comm_proc[n] >= 0) comm_rank[comm_proc[n]] = n;
    for (int i=0;i<64;++i) {
      if ((num_recv_flag & 1<<n) && (recv_cnt_tag[n] & 1L<<i)) {
        comm_recv[r_cnt].cnt = recv_cnt[n][i];
        comm_recv[r_cnt].rank = n;
        comm_recv[r_cnt].dir = i;
        comm_recv[r_cnt].idx = new int[recv_cnt[n][i]];
        for (int j=0,c=0;j<recv_idx[n].size();++j)
          if ((recv_tag[n][j] & (1L<<i)))
            comm_recv[r_cnt].idx[c++] = recv_idx[n][j];
        ++r_cnt;
      }
      if ((num_send_flag & 1<<n) && (send_cnt_tag[n] & 1L<<i)) {
        comm_send[s_cnt].cnt = send_cnt[n][i];
        comm_send[s_cnt].rank = n;
        comm_send[s_cnt].dir = i;
        comm_send[s_cnt].idx = new int[send_cnt[n][i]];
        for (int j=0,c=0;j<send_idx[n].size();++j)
          if ((send_tag[n][j] & 1L<<i))
            comm_send[s_cnt].idx[c++] = send_idx[n][j];
        ++s_cnt;
      }
    }
  }
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_prepare_comm (GhostCommunicator *comm, int data_part) {
  CALL_TRACE();
  prepare_comm(comm, data_part, num_comm_send + num_comm_recv);
  int cnt = 0;
  for (int i=0;i<num_comm_send;++i) {
    comm->comm[cnt].type = GHOST_SEND;
    comm->comm[cnt].node = comm_send[i].rank;
    //comm->comm[cnt].tag = comm_send[i].dir;
    comm->comm[cnt].part_lists = (ParticleList**)Utils::malloc(comm_send[i].cnt*sizeof(ParticleList*));
    comm->comm[cnt].n_part_lists = comm_send[i].cnt;
    for (int n=0;n<comm_send[i].cnt;++n)
      comm->comm[cnt].part_lists[n] = &cells[comm_send[i].idx[n]];
    if (data_part & GHOSTTRANS_POSSHFTD) {
      if ((comm_send[i].dir &  1)) comm->comm[cnt].shift[0] += box_l[0];
      if ((comm_send[i].dir &  2)) comm->comm[cnt].shift[0] -= box_l[0];
      if ((comm_send[i].dir &  4)) comm->comm[cnt].shift[1] += box_l[1];
      if ((comm_send[i].dir &  8)) comm->comm[cnt].shift[1] -= box_l[1];
      if ((comm_send[i].dir & 16)) comm->comm[cnt].shift[2] += box_l[2];
      if ((comm_send[i].dir & 32)) comm->comm[cnt].shift[2] -= box_l[2];
    }
    ++cnt;
  }
  for (int i=0;i<num_comm_recv;++i) {
    comm->comm[cnt].type = GHOST_RECV;
    comm->comm[cnt].node = comm_recv[i].rank;
    /*
    comm->comm[cnt].tag = comm_recv[i].dir;
    if ((comm_recv[i].dir &  3)) comm->comm[cnt].tag ^=  3;
    if ((comm_recv[i].dir & 12)) comm->comm[cnt].tag ^= 12;
    if ((comm_recv[i].dir & 48)) comm->comm[cnt].tag ^= 48;
    */
    comm->comm[cnt].part_lists = (ParticleList**)Utils::malloc(comm_recv[i].cnt*sizeof(ParticleList*));
    comm->comm[cnt].n_part_lists = comm_recv[i].cnt;
    for (int n=0;n<comm_recv[i].cnt;++n)
      comm->comm[cnt].part_lists[n] = &cells[comm_recv[i].idx[n]];
    /*if (data_part & GHOSTTRANS_POSSHFTD) {
      if ((comm_recv[i].dir &  1)) comm->comm[cnt].shift[0] += box_l[0];
      if ((comm_recv[i].dir &  2)) comm->comm[cnt].shift[0] -= box_l[0];
      if ((comm_recv[i].dir &  4)) comm->comm[cnt].shift[1] += box_l[1];
      if ((comm_recv[i].dir &  8)) comm->comm[cnt].shift[1] -= box_l[1];
      if ((comm_recv[i].dir & 16)) comm->comm[cnt].shift[2] += box_l[2];
      if ((comm_recv[i].dir & 32)) comm->comm[cnt].shift[2] -= box_l[2];
    }*/
    ++cnt;
  }
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_mark_cells () {
  CALL_TRACE();
#ifndef P4EST_NOCHANGE
  for (int c=0;c<num_local_cells;++c)
    local_cells.cell[c] = &cells[c];
  for (int c=0;c<num_ghost_cells;++c)
    ghost_cells.cell[c] = &cells[num_local_cells + c];
#endif
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_update_comm_w_boxl(GhostCommunicator *comm) {
  CALL_TRACE();
  int cnt = 0;
  for (int i=0;i<num_comm_send;++i) {
    comm->comm[cnt].shift[0] = comm->comm[cnt].shift[1] = comm->comm[cnt].shift[2] = 0.0;
    if ((comm_send[i].dir &  1)) comm->comm[cnt].shift[0] += box_l[0];
    if ((comm_send[i].dir &  2)) comm->comm[cnt].shift[0] -= box_l[0];
    if ((comm_send[i].dir &  4)) comm->comm[cnt].shift[1] += box_l[1];
    if ((comm_send[i].dir &  8)) comm->comm[cnt].shift[1] -= box_l[1];
    if ((comm_send[i].dir & 16)) comm->comm[cnt].shift[2] += box_l[2];
    if ((comm_send[i].dir & 32)) comm->comm[cnt].shift[2] -= box_l[2];
    ++cnt;
  }
  /*for (int i=0;i<num_comm_recv;++i) {
    comm->comm[cnt].shift[0] = comm->comm[cnt].shift[1] = comm->comm[cnt].shift[2] = 0.0;
    if ((comm_recv[i].dir &  1)) comm->comm[cnt].shift[0] += box_l[0];
    if ((comm_recv[i].dir &  2)) comm->comm[cnt].shift[0] -= box_l[0];
    if ((comm_recv[i].dir &  4)) comm->comm[cnt].shift[1] += box_l[1];
    if ((comm_recv[i].dir &  8)) comm->comm[cnt].shift[1] -= box_l[1];
    if ((comm_recv[i].dir & 16)) comm->comm[cnt].shift[2] += box_l[2];
    if ((comm_recv[i].dir & 32)) comm->comm[cnt].shift[2] -= box_l[2];
    ++cnt;
  }*/
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_init_cell_interaction() {
#ifndef P4EST_NOCHANGE
  dd.cell_inter = (IA_Neighbor_List*)Utils::realloc(dd.cell_inter,local_cells.n*sizeof(IA_Neighbor_List));
  for (int i=0;i<local_cells.n; i++) { 
    dd.cell_inter[i].nList = NULL; 
    dd.cell_inter[i].n_neighbors=0; 
  }
  
  for (int i=0;i<num_local_cells;++i) {
    dd.cell_inter[i].nList = (IA_Neighbor*)Utils::realloc(dd.cell_inter[i].nList,CELLS_MAX_NEIGHBORS*sizeof(IA_Neighbor));
    
    dd.cell_inter[i].nList[0].cell_ind = i;
    dd.cell_inter[i].nList[0].pList = &cells[i];
    init_pairList(&dd.cell_inter[i].n_list[0].vList);
    
    for (int n=1;n<CELLS_MAX_NEIGHBORS;++n) {
      dd.cell_inter[i].nList[n].cell_ind = p4est_shell[i].neighbor[half_neighbor_idx[n]];
      dd.cell_inter[i].nList[n].pList = &cells[p4est_shell[i].neighbor[half_neighbor_idx[n]]];
      init_pairList(&dd.cell_inter[i].n_list[n].vList);
    }
    
    dd.cell_inter[i].n_neighbors = CELLS_MAX_NEIGHBORS;
  }
#endif
}
//--------------------------------------------------------------------------------------------------
Cell* dd_p4est_save_position_to_cell(double pos[3]) {
  CALL_TRACE();
  
  int i;
  int scale_pos[3], scale_pos_l[3], scale_pos_h[3];
  
  for (int d=0;d<3;++d) {
    scale_pos[d] = pos[d]*dd.inv_cell_size[d];
    scale_pos_l[d] = (pos[d] + ROUND_ERROR_PREC*box_l[d])*dd.inv_cell_size[d];
    scale_pos_h[d] = (pos[d] - ROUND_ERROR_PREC*box_l[d])*dd.inv_cell_size[d];
    
    if (!PERIODIC(d) && scale_pos[d] < 0) scale_pos[d] = 0;
    if (!PERIODIC(d) && scale_pos_l[d] < 0) scale_pos_l[d] = 0;
    if (!PERIODIC(d) && scale_pos_h[d] < 0) scale_pos_h[d] = 0;
    if (!PERIODIC(d) && scale_pos[d] >= brick_size[d]) scale_pos[d] = brick_size[d] - 1;
    if (!PERIODIC(d) && scale_pos_l[d] >= brick_size[d]) scale_pos_l[d] = brick_size[d] - 1;
    if (!PERIODIC(d) && scale_pos_h[d] >= brick_size[d]) scale_pos_h[d] = brick_size[d] - 1;
  }
  
  for (i=0;i<num_local_cells;++i) {
    if ((p4est_shell[i].boundary & 1)) {
      if (scale_pos_l[0] < p4est_shell[i].coord[0]) continue;
    } else {
      if (scale_pos[0] < p4est_shell[i].coord[0]) continue;
    }
    if ((p4est_shell[i].boundary & 2)) {
      if (scale_pos_h[0] > p4est_shell[i].coord[0]) continue;
    } else {
      if (scale_pos[0] > p4est_shell[i].coord[0]) continue;
    }
    if ((p4est_shell[i].boundary & 4)) {
      if (scale_pos_l[1] < p4est_shell[i].coord[1]) continue;
    } else {
      if (scale_pos[1] < p4est_shell[i].coord[1]) continue;
    }
    if ((p4est_shell[i].boundary & 8)) {
      if (scale_pos_h[1] > p4est_shell[i].coord[1]) continue;
    } else {
      if (scale_pos[1] > p4est_shell[i].coord[1]) continue;
    }
    if ((p4est_shell[i].boundary & 16)) {
      if (scale_pos_l[0] < p4est_shell[i].coord[0]) continue;
    } else {
      if (scale_pos[0] < p4est_shell[i].coord[0]) continue;
    }
    if ((p4est_shell[i].boundary & 32)) {
      if (scale_pos_h[2] > p4est_shell[i].coord[2]) continue;
    } else {
      if (scale_pos[2] > p4est_shell[i].coord[2]) continue;
    }
    break;
  }
  
  if (i < num_local_cells) &cells[i];
  return NULL;
}
//--------------------------------------------------------------------------------------------------
Cell* dd_p4est_position_to_cell(double pos[3]) {
#ifdef ADDITIONAL_CHECKS
  runtimeErrorMsg() << "function " << __FUNCTION__ << " in " << __FILE__ 
    << "[" << __LINE__ << "] is not implemented";
#endif
  CALL_TRACE();
  
  int i;
  int scale_pos[3];
  
  for (int d=0;d<3;++d) {
    scale_pos[d] = pos[d]*dd.inv_cell_size[d];
    
    if (PERIODIC(d) & scale_pos[d] < 0) scale_pos[d] = 0;
    if (PERIODIC(d) & scale_pos[d] >= brick_size[d]) scale_pos[d] = brick_size[d] - 1;
  }
  
  for (i=0;i<num_local_cells;++i) {
    if (scale_pos[0] != p4est_shell[i].coord[0]) continue;
    if (scale_pos[1] != p4est_shell[i].coord[1]) continue;
    if (scale_pos[2] != p4est_shell[i].coord[2]) continue;
    break;
  }
  
  if (i < num_local_cells) &cells[i];
  return NULL;
}
//--------------------------------------------------------------------------------------------------
int dd_p4est_position_to_cell_all(double pos[3], int guess = -1) {
  CALL_TRACE();
  
  int i;
  int scale_pos[3];
  
  for (int d=0;d<3;++d) {
    scale_pos[d] = pos[d]*dd.inv_cell_size[d];
    
    if (PERIODIC(d) & scale_pos[d] < 0) scale_pos[d] = 0;
    if (PERIODIC(d) & scale_pos[d] >= brick_size[d]) scale_pos[d] = brick_size[d] - 1;
  }
  
  if (guess >= 0 && guess < num_cells) {
    int flag = 1;
    if (scale_pos[0] != p4est_shell[guess].coord[0]) flag = 0;
    if (scale_pos[1] != p4est_shell[guess].coord[1]) flag = 0;
    if (scale_pos[2] != p4est_shell[guess].coord[2]) flag = 0;
    return guess;
  }
  
  for (i=0;i<num_cells;++i) {
    if (scale_pos[0] != p4est_shell[i].coord[0]) continue;
    if (scale_pos[1] != p4est_shell[i].coord[1]) continue;
    if (scale_pos[2] != p4est_shell[i].coord[2]) continue;
    break;
  }
  
  if (i < num_cells) return i;
  return -1;
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_position_to_cell(double pos[3], int* idx) {
  runtimeErrorMsg() << "function " << __FUNCTION__ << " in " << __FILE__ 
    << "[" << __LINE__ << "] is not implemented";
}
//--------------------------------------------------------------------------------------------------
void dd_p4est_exchange_and_sort_particles(int global_flag) {
  ParticleList send_buf[num_comm_proc], recv_buf[num_comm_proc];
  
  for (int i=0;i<num_comm_proc;++i) {
    init_particlelist(&send_buf[i]);
    init_particlelist(&recv_buf[i]);
  }
  
  // start recv
  
  for (int i=0;i<num_local_cells;++i) {
    Cell* cell = local_cells.cell[i];
    local_shell_t* shell = &p4est_shell[i];
    
    double cell_lc[3], cell_hc[3];
    for (int d=0;d<3;++d) {
      cell_lc[d] = dd.cell_size[d]*shell->coord[d];
      cell_hc[d] = dd.cell_size[d]*(shell->coord[d] + 1);
      if ((shell->boundary & (1<<(2*d)))) cell_lc[d] -= 0.5*ROUND_ERROR_PREC*box_l[d];
      if ((shell->boundary & (2<<(2*d)))) cell_hc[d] += 0.5*ROUND_ERROR_PREC*box_l[d];
    }
    
    for (int p=0;p<cell->n;++p) {
      Particle* part = &cell->part[p];
      int x,y,z;
      
      if (part->r.p[0] < cell_lc[0]) x = 0;
      else if (part->r.p[0] >= cell_hc[0]) x = 2;
      else x = 1;
      if (part->r.p[1] < cell_lc[1]) y = 0;
      else if (part->r.p[1] >= cell_hc[1]) y = 2;
      else y = 1;
      if (part->r.p[2] < cell_lc[2]) z = 0;
      else if (part->r.p[2] >= cell_hc[2]) z = 2;
      else z = 1;
      
      int nidx = neighbor_lut[z][y][x];
      if (nidx != -1) { // Particle p outside of cell i
        nidx = shell->neighbor[nidx];
        if (nidx >= num_local_cells) { // Remote Cell
          if (p4est_shell[nidx].rank >= 0) {
            CELL_TRACE(fprintf(stderr,"%d: dd_ex_and_sort_p: send part %d\n",this_node,part->p.identity));
            local_particles[part->p.identity] = NULL;
            if ((p4est_shell[nidx].boundary &  3))
              fold_coordinate(part->r.p, part->m.v, part->l.i, 0);
            if ((p4est_shell[nidx].boundary & 12))
              fold_coordinate(part->r.p, part->m.v, part->l.i, 1);
            if ((p4est_shell[nidx].boundary & 48)) 
              fold_coordinate(part->r.p, part->m.v, part->l.i, 2);
            move_indexed_particle(&send_buf[comm_proc[p4est_shell[nidx].rank]], cell, p);
            if(p < cell->n) p -= 1;
          }
        } else { // Local Cell
          move_indexed_particle(&cells[nidx], cell, p);
          if(p < cell->n) p -= 1;
        }
      }
    }
  }
  
  // send
  for (int i=0;i<num_comm_proc;++i) {
    if (comm_rank[i] == this_node) {
      for (int p=0;p<send_buf[i].n;++p) {
        Cell* target = dd_p4est_save_position_to_cell(send_buf[i].part[p].r.p);
        if (target)
          append_indexed_particle(target, &send_buf[i].part[p]);
        else
          runtimeErrorMsg() << this_node << " received local particle out of domain";
      }
    } else {
      // send to other proc
    }
  }
    
  // wait for recv
  
  // append
  for (int i=0;i<num_comm_proc;++i) {
    if (comm_rank[i] != this_node) {
      for (int p=0;p<recv_buf[i].n;++p) {
        Cell* target = dd_p4est_save_position_to_cell(recv_buf[i].part[p].r.p);
        if (target) {
          append_indexed_particle(target, &recv_buf[i].part[p]);
        } else {
          runtimeErrorMsg() << this_node << " received remote particle out of domain";
        }
      }
    }
  }
}
//--------------------------------------------------------------------------------------------------
#endif
