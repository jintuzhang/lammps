/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributors
      William C Witt (University of Cambridge)
------------------------------------------------------------------------- */

#include "pair_mace_kokkos.h"

// TODO: do i need all these?
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list.h"
#include "neighbor_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairMACEKokkos<DeviceType>::PairMACEKokkos(LAMMPS *lmp) : PairMACE(lmp)
{
std::cout << "hello from kokkos mace constructor" << std::endl;
  no_virial_fdotr_compute = 1;

  //kokkosable = 1;
  //atomKK = (AtomKokkos *) atom;
  //execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  //datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  //datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
  // pair_pace_kokkos has these instead
  //datamask_read = EMPTY_MASK;
  //datamask_modify = EMPTY_MASK;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

  host_flag = (execution_space == Host);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairMACEKokkos<DeviceType>::~PairMACEKokkos()
{
  if (copymode) return;

  // from lj_cut_kokkos
  //if (allocated) {
  //  memoryKK->destroy_kokkos(k_eatom,eatom);
  //  memoryKK->destroy_kokkos(k_vatom,vatom);
  //  memoryKK->destroy_kokkos(k_cutsq,cutsq);
  //}

  // from pair_pace_kokkos
  //memoryKK->destroy_kokkos(k_eatom,eatom);
  //memoryKK->destroy_kokkos(k_vatom,vatom);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMACEKokkos<DeviceType>::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag,0);

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK|TAG_MASK);

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;

  if (atom->nlocal != list->inum) error->all(FLERR, "ERROR: nlocal != inum.");
  if (domain_decomposition) {
    if (atom->nghost != list->gnum) error->all(FLERR, "ERROR: nghost != gnum.");
  }

  if (eflag_atom || vflag_atom) {
    error->all(FLERR, "ERROR: kokkos eflag_atom not implemented.");
  }

//  if (eflag_atom) {
//    memoryKK->destroy_kokkos(k_eatom,eatom);
//    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
//    d_eatom = k_eatom.view<DeviceType>();
//  }
//  if (vflag_atom) {
//    memoryKK->destroy_kokkos(k_vatom,vatom);
//    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
//    d_vatom = k_vatom.view<DeviceType>();
//  }

  int nlocal = atom->nlocal;
  auto r_max_squared = this->r_max_squared;
  auto h0 = domain->h[0];
  auto h1 = domain->h[1];
  auto h2 = domain->h[2];
  auto h3 = domain->h[3];
  auto h4 = domain->h[4];
  auto h5 = domain->h[5];
  auto hinv0 = domain->h_inv[0];
  auto hinv1 = domain->h_inv[1];
  auto hinv2 = domain->h_inv[2];
  auto hinv3 = domain->h_inv[3];
  auto hinv4 = domain->h_inv[4];
  auto hinv5 = domain->h_inv[5];

  auto k_lammps_atomic_numbers = Kokkos::View<int64_t*,DeviceType>("k_lammps_atomic_numbers",lammps_atomic_numbers.size());
  auto k_lammps_atomic_numbers_mirror = Kokkos::create_mirror_view(k_lammps_atomic_numbers);
  for (int i=0; i<lammps_atomic_numbers.size(); ++i) {
    k_lammps_atomic_numbers_mirror(i) = lammps_atomic_numbers[i];
  }
  auto k_mace_atomic_numbers = Kokkos::View<int64_t*,DeviceType>("k_mace_atomic_numbers",mace_atomic_numbers.size());
  auto k_mace_atomic_numbers_mirror = Kokkos::create_mirror_view(k_mace_atomic_numbers);
  for (int i=0; i<lammps_atomic_numbers.size(); ++i) {
    k_mace_atomic_numbers_mirror(i) = mace_atomic_numbers[i];
  }
  auto mace_atomic_numbers_size = mace_atomic_numbers.size();

  // atom map
  auto map_style = atom->map_style;
  auto k_map_array = atomKK->k_map_array;
  auto k_map_hash = atomKK->k_map_hash;
  k_map_array.template sync<DeviceType>();


  auto x = atomKK->k_x.view<DeviceType>();
//  c_x = atomKK->k_x.view<DeviceType>();
  auto f = atomKK->k_f.view<DeviceType>();
  auto tag = atomKK->k_tag.view<DeviceType>();
  auto type = atomKK->k_type.view<DeviceType>();


  // ----- positions -----
  int n_nodes;
  if (domain_decomposition) {
    n_nodes = atom->nlocal + atom->nghost;
  } else {
    // normally, ghost atoms are included in the graph as independent
    // nodes, as required when the local domain does not have PBC.
    // however, in no_domain_decomposition mode, ghost atoms are known to
    // be shifted versions of local atoms.
    n_nodes = atom->nlocal;
  }
  auto k_positions = Kokkos::View<double*[3],Kokkos::LayoutRight,DeviceType>("k_positions", n_nodes);
  Kokkos::parallel_for(n_nodes, KOKKOS_LAMBDA (const int i) {
    k_positions(i,0) = x(i,0);
    k_positions(i,1) = x(i,1);
    k_positions(i,2) = x(i,2);
  });
  auto positions = torch::from_blob(
    k_positions.data(),
    {n_nodes,3},
    torch::TensorOptions().dtype(torch_float_dtype).device(device));

  // ----- cell -----
  // TODO: how to use kokkos here?
  auto cell = torch::zeros({3,3}, torch::TensorOptions().dtype(torch_float_dtype).device(device));
  cell[0][0] = h0;
  cell[0][1] = 0.0;
  cell[0][2] = 0.0;
  cell[1][0] = h5;
  cell[1][1] = h1;
  cell[1][2] = 0.0;
  cell[2][0] = h4;
  cell[2][1] = h3;
  cell[2][2] = h2;

  // ----- edge_index and unit_shifts -----
  // count total number of edges
  auto k_n_edges_vec = Kokkos::View<int64_t*,DeviceType>("k_n_edges_vec", n_nodes);
  Kokkos::parallel_for(n_nodes, KOKKOS_LAMBDA (const int ii) {
    const int i = d_ilist(ii);
    const double xtmp = x(i,0);
    const double ytmp = x(i,1);
    const double ztmp = x(i,2);
    for (int jj=0; jj<d_numneigh(i); ++jj) {
      int j = d_neighbors(i,jj);
      j &= NEIGHMASK;
      const double delx = xtmp - x(j,0);
      const double dely = ytmp - x(j,1);
      const double delz = ztmp - x(j,2);
      const double rsq = delx*delx + dely*dely + delz*delz;
      if (rsq < r_max_squared) {
        k_n_edges_vec(ii) += 1;
      }
    }
  });
  // WARNING: if n_edges remains 0 (e.g., because atoms too far apart)
  // strange things happen on gpu
  int64_t n_edges = 0;
  Kokkos::parallel_reduce(n_nodes, KOKKOS_LAMBDA(const int ii, int64_t& n_edges) {
    n_edges += k_n_edges_vec(ii);
  }, n_edges);
  // make first_edge vector to help with parallelizing following loop
  auto k_first_edge = Kokkos::View<int64_t*,DeviceType>("k_first_edge", n_nodes);  // initialized to zero
  Kokkos::parallel_for(n_nodes-1, KOKKOS_LAMBDA(const int ii) {
    k_first_edge(ii+1) = k_first_edge(ii) + k_n_edges_vec(ii);
  });
  // fill edge_index and unit_shifts tensors
  auto k_edge_index = Kokkos::View<int64_t**,Kokkos::LayoutRight,DeviceType>("k_edge_index", 2, n_edges);
  auto k_unit_shifts = Kokkos::View<double*[3],Kokkos::LayoutRight,DeviceType>("k_unit_shifts", n_edges);
  auto k_shifts = Kokkos::View<double*[3],Kokkos::LayoutRight,DeviceType>("k_shifts", n_edges);

  if (domain_decomposition) {

    Kokkos::parallel_for(n_nodes, KOKKOS_LAMBDA(const int ii) {
      const int i = d_ilist(ii);
      const double xtmp = x(i,0);
      const double ytmp = x(i,1);
      const double ztmp = x(i,2);
      int k = k_first_edge(ii);
      for (int jj=0; jj<d_numneigh(i); ++jj) {
        int j = d_neighbors(i,jj);
        j &= NEIGHMASK;
        const double delx = xtmp - x(j,0);
        const double dely = ytmp - x(j,1);
        const double delz = ztmp - x(j,2);
        const double rsq = delx*delx + dely*dely + delz*delz;
        if (rsq < r_max_squared) {
          k_edge_index(0,k) = i;
          k_edge_index(1,k) = j;
          k++;
        }
      }
    });

  } else {

    Kokkos::parallel_for(n_nodes, KOKKOS_LAMBDA(const int ii) {
      const int i = d_ilist(ii);
      const double xtmp = x(i,0);
      const double ytmp = x(i,1);
      const double ztmp = x(i,2);
      int k = k_first_edge(ii);
      for (int jj=0; jj<d_numneigh(i); ++jj) {
        int j = d_neighbors(i,jj);
        j &= NEIGHMASK;
        const double delx = xtmp - x(j,0);
        const double dely = ytmp - x(j,1);
        const double delz = ztmp - x(j,2);
        const double rsq = delx*delx + dely*dely + delz*delz;
        if (rsq < r_max_squared) {
          k_edge_index(0,k) = i;
          //int j_local = atom->map(tag(j));
          int j_local = AtomKokkos::map_kokkos<DeviceType>(tag(j),map_style,k_map_array,k_map_hash);
          k_edge_index(1,k) = j_local;
          double shiftx = x(j,0) - x(j_local,0);
          double shifty = x(j,1) - x(j_local,1);
          double shiftz = x(j,2) - x(j_local,2);
          double shiftxs = std::round(hinv0*shiftx + hinv5*shifty + hinv4*shiftz);
          double shiftys = std::round(hinv1*shifty + hinv3*shiftz);
          double shiftzs = std::round(hinv2*shiftz);
          k_unit_shifts(k,0) = shiftxs;
          k_unit_shifts(k,1) = shiftys;
          k_unit_shifts(k,2) = shiftzs;
          k_shifts(k,0) = h0*shiftxs + h5*shiftys + h4*shiftzs;
          k_shifts(k,1) = h1*shiftys + h3*shiftzs;
          k_shifts(k,2) = h2*shiftzs;
          k++;
        }
      }
    });
  }
  auto edge_index = torch::from_blob(
    k_edge_index.data(),
    {2,n_edges},
    torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto unit_shifts = torch::from_blob(
    k_unit_shifts.data(),
    {n_edges,3},
    torch::TensorOptions().dtype(torch_float_dtype).device(device));
  auto shifts = torch::from_blob(
    k_shifts.data(),
    {n_edges,3},
    torch::TensorOptions().dtype(torch_float_dtype).device(device));

  // ----- node_attrs -----
  // node_attrs is one-hot encoding for atomic numbers
  int n_node_feats = mace_atomic_numbers_size;
  auto k_node_attrs = Kokkos::View<double**,Kokkos::LayoutRight,DeviceType>("k_node_attrs", n_nodes, n_node_feats);
  Kokkos::parallel_for(n_nodes, KOKKOS_LAMBDA(const int ii) {
    const int i = d_ilist(ii);
    const int lammps_type = type(i);
    int t = -1;
    for (int j=0; j<mace_atomic_numbers_size; ++j) {
      if (k_mace_atomic_numbers(j)==k_lammps_atomic_numbers(lammps_type-1)) {
        t = j+1;
      }
    }
    //if (t==-1) error->all(FLERR, "ERROR: problem converting lammps_type to mace_type.");
    k_node_attrs(i,t-1) = 1.0;
  });
  auto node_attrs = torch::from_blob(
    k_node_attrs.data(),
    {n_nodes, n_node_feats},
    torch::TensorOptions().dtype(torch_float_dtype).device(device));

  // ----- mask for ghost -----
  Kokkos::View<bool*,DeviceType> k_mask("k_mask", n_nodes);
  Kokkos::parallel_for(nlocal, KOKKOS_LAMBDA(const int ii) {
    const int i = d_ilist(ii);
    k_mask(i) = true;
  });
  auto mask = torch::from_blob(
    k_mask.data(),
    n_nodes,
    torch::TensorOptions().dtype(torch::kBool).device(device));

  // TODO: why is batch of size n_nodes?
  // TODO: add device?
  auto batch = torch::zeros({n_nodes}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto energy = torch::empty({1}, torch::TensorOptions().dtype(torch_float_dtype).device(device));
  auto forces = torch::empty({n_nodes,3}, torch::TensorOptions().dtype(torch_float_dtype).device(device));
  auto ptr = torch::empty({2}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto weight = torch::empty({1}, torch::TensorOptions().dtype(torch_float_dtype).device(device));
  ptr[0] = 0;
  ptr[1] = n_nodes;
  weight[0] = 1.0;

  // pack the input, call the model, extract the output
  c10::Dict<std::string, torch::Tensor> input;
  input.insert("batch", batch);
  input.insert("cell", cell);
  input.insert("edge_index", edge_index);
  input.insert("energy", energy);
  input.insert("forces", forces);
  input.insert("node_attrs", node_attrs);
  input.insert("positions", positions);
  input.insert("ptr", ptr);
  input.insert("shifts", shifts);
  input.insert("unit_shifts", unit_shifts);
  input.insert("weight", weight);
std::cout << "batch" << batch.to("cpu") << std::endl;
std::cout << "cell" << cell.to("cpu") << std::endl;
//std::cout << "edge_index" << edge_index.to("cpu") << std::endl;
// energy
// forces
//std::cout << "node_attrs" << node_attrs.to("cpu") << std::endl;
std::cout << "positions" << positions.to("cpu") << std::endl;
std::cout << "ptr" << ptr.to("cpu") << std::endl;
std::cout << "shifts" << shifts.to("cpu") << std::endl;
std::cout << "unit_shifts" << unit_shifts.to("cpu") << std::endl;
std::cout << "weight" << weight.to("cpu") << std::endl;
std::cout << "mask" << mask.to("cpu") << std::endl;
  auto output = model.forward({input, mask, true, true, false}).toGenericDict();

  //std::cout << "node energy: " << typeid(output.at("node_energy").toTensor()).name() << std::endl;
  //std::cout << "node energy: " << typeid(output.at("node_energy").toTensor().data_ptr()).name() << std::endl;
  //std::cout << "forces: " << typeid(output.at("forces").toTensor()).name() << std::endl;
  //std::cout << "forces: " << typeid(output.at("forces").toTensor().data_ptr()).name() << std::endl;


  // mace energy
  //   -> sum of site energies of local atoms
  if (eflag_global) {
    auto node_energy = output.at("node_energy").toTensor();
    auto node_energy_ptr = static_cast<double*>(node_energy.data_ptr());
    auto k_node_energy = Kokkos::View<double*,Kokkos::LayoutRight,DeviceType,Kokkos::MemoryTraits<Kokkos::Unmanaged>>(node_energy_ptr,n_nodes);
    eng_vdwl = 0.0;
    Kokkos::parallel_reduce(nlocal, KOKKOS_LAMBDA(const int ii, double &eng_vdwl) {
      const int i = d_ilist(ii);
      eng_vdwl += k_node_energy(i);
    }, eng_vdwl);
  }

  // mace forces
  //   -> derivatives of total mace energy
  forces = output.at("forces").toTensor();
  auto forces_ptr = static_cast<double*>(forces.data_ptr());
  auto k_forces = Kokkos::View<double*[3],Kokkos::LayoutRight,DeviceType,Kokkos::MemoryTraits<Kokkos::Unmanaged>>(forces_ptr,n_nodes);
  Kokkos::parallel_for(nlocal, KOKKOS_LAMBDA(const int ii) {
    const int i = d_ilist(ii);
    f(i,0) = k_forces(i,0);
    f(i,1) = k_forces(i,1);
    f(i,2) = k_forces(i,2);
  });
//
////  // mace site energies
////  //   -> local atoms only
////  if (eflag_atom) {
////    auto node_energy = output.at("node_energy").toTensor();
////    #pragma omp parallel for
////    for (int ii=0; ii<list->inum; ++ii) {
////      int i = list->ilist[ii];
////      eatom[i] = node_energy[i].item<double>();
////    }
////  }
////
//  // mace virials (local atoms only)
//  //   -> derivatives of sum of site energies of local atoms
//  if (vflag_global) {
//    auto vir = output.at("virials").toTensor();
//    virial[0] = vir[0][0][0].item<double>();
//    virial[1] = vir[0][1][1].item<double>();
//    virial[2] = vir[0][2][2].item<double>();
//    virial[3] = 0.5*(vir[0][2][1].item<double>() + vir[0][1][2].item<double>());
//    virial[4] = 0.5*(vir[0][2][0].item<double>() + vir[0][0][2].item<double>());
//    virial[5] = 0.5*(vir[0][1][0].item<double>() + vir[0][0][1].item<double>());
//  }
////
////  // mace site virials
////  //   -> not available
////  if (vflag_atom) {
////    error->all(FLERR, "ERROR: pair_mace does not support vflag_atom.");
////  }
//
//
//  //copymode = 1;
//
//  //EV_FLOAT ev = pair_compute<PairLJCutKokkos<DeviceType>,void >(this,(NeighListKokkos<DeviceType>*)list);
//
//  //if (eflag_global) eng_vdwl += ev.evdwl;
//  //if (vflag_global) {
//  //  virial[0] += ev.v[0];
//  //  virial[1] += ev.v[1];
//  //  virial[2] += ev.v[2];
//  //  virial[3] += ev.v[3];
//  //  virial[4] += ev.v[4];
//  //  virial[5] += ev.v[5];
//  //}
//
//  //if (eflag_atom) {
//  //  k_eatom.template modify<DeviceType>();
//  //  k_eatom.template sync<LMPHostType>();
//  //}
//
//  //if (vflag_atom) {
//  //  k_vatom.template modify<DeviceType>();
//  //  k_vatom.template sync<LMPHostType>();
//  //}
//
//  copymode = 0;

std::cout << "energy: " << output.at("energy").toTensor().to("cpu") << std::endl;
std::cout << "node_energy: " << output.at("node_energy").toTensor().to("cpu") << std::endl;

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMACEKokkos<DeviceType>::coeff(int narg, char **arg)
{
std::cout << "hello from kokkos coeff" << std::endl;
  if (!allocated) allocate();
std::cout << "allocated: " << allocated << std::endl;
  PairMACE::coeff(narg,arg);
std::cout << "allocated: " << allocated << std::endl;
}

template<class DeviceType>
void PairMACEKokkos<DeviceType>::init_style()
{
std::cout << "hello from kokkos init_style" << std::endl;
//  if (force->newton_pair == 0) error->all(FLERR, "ERROR: Pair style mace requires newton pair on.");

//  /*
//    MACE requires the full neighbor list AND neighbors of ghost atoms
//    it appears that:
//      * without REQ_GHOST
//           list->gnum == 0
//           list->ilist does not include ghost atoms, but the jlists do
//      * with REQ_GHOST
//           list->gnum == atom->nghost
//           list->ilist includes ghost atoms
//  */
//  if (domain_decomposition) {
//    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
//  } else {
//    neighbor->add_request(this, NeighConst::REQ_FULL);
//  }

//  if (host_flag) {
//    PairMACE::init_style();
//    return;
//  }
//std::cout << "after mace init" << std::endl;
//
//  // neighbor list request for KOKKOS
//
//  neighflag = lmp->kokkos->neighflag;
//
//  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
//  //if (domain_decomposition) {
//    //auto request = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
//    request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
//                             !std::is_same<DeviceType,LMPDeviceType>::value);
//    request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);
//    if (neighflag == FULL) request->enable_full();
  //} else {
  //  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  //  request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
  //                           !std::is_same<DeviceType,LMPDeviceType>::value);
  //  request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);
  //}
//  if (neighflag == FULL)
//    error->all(FLERR,"Must use half neighbor list style with pair pace/kk");





  PairMACE::init_style();
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
                           !std::is_same<DeviceType,LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);






//  // TODO: from lj_cut, possibly delete
//  // adjust neighbor list request for KOKKOS
//  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
//  request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
//                           !std::is_same<DeviceType,LMPDeviceType>::value);
//  request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);
//
//  neighflag = lmp->kokkos->neighflag;
//  auto request = neighbor->find_request(this);
//std::cout << "before request" << std::endl;
//  request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
//                           !std::is_same<DeviceType,LMPDeviceType>::value);
//std::cout << "before request" << std::endl;
//  request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);
//std::cout << "after request" << std::endl;
//  if (neighflag == FULL)
//    std::cout << "REQUESTING FULL LIST." << std::endl;
//  if (neighflag == FULL) request->enable_full();

  //auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  //request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
  //                         !std::is_same<DeviceType,LMPDeviceType>::value);
  //request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);
  //if (neighflag == FULL)
  //  error->all(FLERR,"Must use half neighbor list style with pair pace/kk");

std::cout << "goodbye from init_style" << std::endl;
}

template<class DeviceType>
double PairMACEKokkos<DeviceType>::init_one(int i, int j)
{
std::cout << "hello from kokkos init_one" << std::endl;
  double cutone = PairMACE::init_one(i,j);

  //k_scale.h_view(i,j) = k_scale.h_view(j,i) = scale[i][j];
  //k_scale.template modify<LMPHostType>();

  k_cutsq.h_view(i,j) = k_cutsq.h_view(j,i) = cutone*cutone;
  k_cutsq.template modify<LMPHostType>();

  return cutone;

  //// to account for message passing, require cutoff of n_layers * r_max
  //return num_interactions*model.attr("r_max").toTensor().item<double>();

//  k_params.h_view(i,j).lj1 = lj1[i][j];
//  k_params.h_view(i,j).lj2 = lj2[i][j];
//  k_params.h_view(i,j).lj3 = lj3[i][j];
//  k_params.h_view(i,j).lj4 = lj4[i][j];
//  k_params.h_view(i,j).offset = offset[i][j];
//  k_params.h_view(i,j).cutsq = cutone*cutone;
//  k_params.h_view(j,i) = k_params.h_view(i,j);
//  if (i<MAX_TYPES_STACKPARAMS+1 && j<MAX_TYPES_STACKPARAMS+1) {
//    m_params[i][j] = m_params[j][i] = k_params.h_view(i,j);
//    m_cutsq[j][i] = m_cutsq[i][j] = cutone*cutone;
//  }
//
//  k_cutsq.h_view(i,j) = k_cutsq.h_view(j,i) = cutone*cutone;
//  k_cutsq.template modify<LMPHostType>();
//  k_params.template modify<LMPHostType>();

//  return cutone;
}

template<class DeviceType>
void PairMACEKokkos<DeviceType>::allocate()
{
std::cout << "hello from kokkos allocate" << std::endl;
  PairMACE::allocate();

  int n = atom->ntypes + 1;
  //MemKK::realloc_kokkos(d_map, "pace:map", n);

  MemKK::realloc_kokkos(k_cutsq, "mace:cutsq", n, n);
  d_cutsq = k_cutsq.template view<DeviceType>();

  // TODO: from lj_cut, possibly delete
  //int n = atom->ntypes;
  //memory->destroy(cutsq);
  //memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  //d_cutsq = k_cutsq.template view<DeviceType>();
  //k_params = Kokkos::DualView<params_lj**,Kokkos::LayoutRight,DeviceType>("PairLJCut::params",n+1,n+1);
  //params = k_params.template view<DeviceType>();
}

namespace LAMMPS_NS {
template class PairMACEKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairMACEKokkos<LMPHostType>;
#endif
}

