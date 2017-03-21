cellsystem domain_decomposition -no_verlet_list
 
set n_part 20
set iter 200
set iteri 200

setmd skin 0.1

inter 0 0 lennard-jones 1.0 0.1 0.8 auto 0.0
 
set boxl 4.0

setmd box_l $boxl $boxl 1

#part 0 pos [expr $boxl*0.5] [expr $boxl*0.5] 0.5 q 0.0 type 0 v 0 0 1 ext_force 0 0 0
#part 0 pos 1.162820 2.642298 0.1 q 0.0 type 0 v 0 0 -1 ext_force 0 0 0

set agrid [expr 1./pow(2, 3)]

expr srand([pid])
for {set i 1} {$i <= $n_part} {incr i} {
  set pos_x [expr rand()*($boxl-2*$agrid) + $agrid]
  set pos_y [expr rand()*($boxl-2*$agrid) + $agrid]
  set pos_z [expr rand()*1.0]
  part $i pos $pos_x $pos_y $pos_z q 0.0 type 0 ext_force 0 0 0
}

set dt 0.0005

thermostat lb 1
setmd time_step $dt

puts "Mindist: [analyze mindist]"
minimize_energy 10.0 100 0.1 0.01
kill_particle_motion
kill_particle_forces
puts "Mindist: [analyze mindist]"

lbadapt-init 1
lbadapt_set_max_level 3

lbfluid cpu agrid $agrid dens 1. visc 0.032 tau $dt friction 0.5 ext_force 0.01 0.0 0

# set inflow/outflow boundaries 
#lbboundary wall dist $agrid normal 1 0 0  velocity 1 0 0 
#lbboundary wall dist [expr -$boxl + $agrid] normal -1 0 0 velocity 1 0 0
# setup a channel 
lbboundary wall dist $agrid normal 0 1 0 
lbboundary wall dist [expr -$boxl + $agrid] normal 0 -1 0
#lbboundary wall dist 0.5 normal 0 1 0
#lbboundary wall dist [expr $boxl -0.5] normal 0 -1 0

#lbadapt-exclude-bnd-from-geom-ref 0
#lbadapt-exclude-bnd-from-geom-ref 1
lbadapt-geom-ref

lbfluid print vtk boundary boundary.vtk

for { set i 0 } { $i < $iter } { incr i } {
  md_vtk "md_part_$i.vtk"
  lbfluid print vtk velocity  "lb_field_$i.vtk"
  integrate $iteri
  puts -nonewline "run $i at time=[setmd time]\r"
  flush stdout
}

lbfluid print vtk velocity  "lb_field_$iter.vtk"
md_vtk "md_part_$iter.vtk"

exit
