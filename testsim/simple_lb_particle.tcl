cellsystem domain_decomposition -no_verlet_list

set n_part 0
set iter 200
set iteri 50

setmd skin 0.1

inter 0 0 lennard-jones 1.0 0.1 0.8 auto 0.0

set boxl 4.0

setmd box_l $boxl $boxl 1

#part 0 pos 1.9 1.9 0.5 q 0.0 type 0 v 0.1 0 0 ext_force 0.00 0 0 fix 1 1 1
part 0 pos 2.0 1.9 0.5 q 0.0 type 0 v 0 0 0 ;#ext_force 0.01 0 0

set base 3
set mlvl 3
set wall 0.5

set agrid [expr 1./pow(2, $mlvl)]

expr srand([pid])
for {set i 1} {$i <= $n_part} {incr i} {
  set pos_x [expr rand()*($boxl-2*$wall) + $wall]
  set pos_y [expr rand()*($boxl-2*$wall) + $wall]
  set pos_z [expr rand()*1.0]
  part $i pos $pos_x $pos_y $pos_z q 0.0 type 0 ext_force 0.1 0 0
}

set dt 0.01

set prefac [expr pow(2, $mlvl - 1)]

thermostat lb 0
setmd time_step [expr $dt/$prefac]

#puts "Mindist: [analyze mindist]"
#minimize_energy 10.0 100 0.1 0.01
#kill_particle_motion
#kill_particle_forces
#puts "Mindist: [analyze mindist]"

lbadapt-init $base
lbadapt_set_max_level $mlvl

lbfluid cpu agrid $agrid dens 1. visc 0.032 tau [expr $dt/$prefac] friction 0.5 ext_force 0.01 0.0 0.0

lbboundary wall dist $wall normal 0 1 0 
lbboundary wall dist [expr -$boxl + $wall] normal 0 -1 0 

#lbadapt-geom-ref

#lbfluid print vtk boundary boundary.vtk

lbadapt-reset-fluid

lbfluid print vtk velocity  lb_0.vtk
md_vtk md_0.vtk

for { set i 0 } { $i < $iter } { incr i } {
  #md_vtk "md_part_$i.vtk"
  #lbfluid print vtk velocity  "lb_field_$i.vtk"
  integrate [expr $iteri*int($prefac)]
  puts -nonewline "run $i at time=[setmd time]\r"
  flush stdout
}

#lbfluid print vtk velocity  "lb_field_$iter.vtk"
lbfluid print vtk velocity  lb_1.vtk
#md_vtk "md_part_$iter.vtk"
md_vtk md_1.vtk

exit
