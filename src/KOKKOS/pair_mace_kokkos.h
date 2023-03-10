/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mace/kk,PairMACEKokkos<LMPDeviceType>);
PairStyle(mace/kk/device,PairMACEKokkos<LMPDeviceType>);
PairStyle(mace/kk/host,PairMACEKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_MACE_KOKKOS_H
#define LMP_PAIR_MACE_KOKKOS_H

#include "pair_mace.h"
#include "kokkos_type.h"
#include "pair_kokkos.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairMACEKokkos : public PairMACE {

 public:

  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairMACEKokkos(class LAMMPS *);
  ~PairMACEKokkos() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void allocate();

 protected:

  bool domain_decomposition = true;
  torch::jit::script::Module model;
  torch::ScalarType torch_float_dtype;
  double r_max;
  double r_max_squared;
  int64_t num_interactions;
  std::vector<int64_t> mace_atomic_numbers;
  std::vector<int64_t> lammps_atomic_numbers;
  const std::array<std::string,118> periodic_table =
    { "H", "He",
     "Li", "Be",                                                              "B",  "C",  "N",  "O",  "F", "Ne",
     "Na", "Mg",                                                             "Al", "Si",  "P",  "S", "Cl", "Ar",
     "K",  "Ca", "Sc", "Ti",  "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
     "Rb", "Sr",  "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",  "I", "Xe",
     "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                       "Hf", "Ta",  "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
     "Fr", "Ra", "Ac", "Th", "Pa",  "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
                       "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};


  // kokkos stuff
  int host_flag;
  int neighflag;
  //typename AT::t_x_array_randomread x;
  typename AT::t_x_array c_x;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread type;

  
  //Kokkos::View<double*[3], DeviceType> k_positions;

  typedef Kokkos::DualView<F_FLOAT**, DeviceType> tdual_fparams;
  tdual_fparams k_cutsq;
  typedef Kokkos::View<F_FLOAT**, DeviceType> t_fparams;
  t_fparams d_cutsq;

};
}    // namespace LAMMPS_NS

#endif
#endif
