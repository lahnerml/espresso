proc rescale_velocities { target_temperature particle_number } {
        set energies [analyze energy]
        set kinetic [lindex $energies 1 1]
        set factor [expr sqrt(0.5*$target_temperature*(3.0*$particle_number-3.0)/$kinetic)]
        for {set i 0} {$i<$particle_number} {incr i} {
                set vel [part $i print v]
                part $i v [expr [lindex $vel 0]*$factor] [expr [lindex $vel 1]*$factor] [expr [lindex $vel 2]*$factor]
        }
}

cellsystem domain_decomposition -no_verlet_list
setmd skin 0.1

set n_part 1000
set density 0.8442
set min_dist 0.87
set boxl [expr pow($n_part/$density,1.0/3.0)+2*0.1]
setmd box_l $boxl $boxl $boxl

inter 0 0 lennard-jones 1.0 1.0 2.5 auto 0.0
thermostat off

if { [file exists particle.dat]==0 } {
  expr srand([pid])
  for {set i 0} {$i < $n_part} {incr i} {
    set pos_x [expr rand()*$boxl]
    set pos_y [expr rand()*$boxl]
    set pos_z [expr rand()*$boxl]
    part $i pos $pos_x $pos_y $pos_z q 0.0 type 0
  }

  thermostat off
  setmd time_step 0.001
  puts "Mindist: [analyze mindist]"
  set cap 1.0
  inter forcecap $cap
  set i 0
  set act_min_dist [analyze mindist]
  while { $i < 2000 && $act_min_dist < $min_dist } {    
      integrate 100

      set act_min_dist [analyze mindist]

      set cap [expr $cap+1.0]
      inter forcecap $cap
      incr i
  }
  inter forcecap 0
  puts "Mindist: [analyze mindist]"

  set target_temperature 0.728
  setmd time_step 0.0001
  for { set n 0 } { $n < 200 } { incr n } {
    integrate 1000
    rescale_velocities  $target_temperature [setmd n_part]
  }
  
  set out [open "particle.dat" "w"]
  blockfile $out write particles "id pos v f type" "all"
  close $out
} else {
  set in [open "particle.dat" "r"]
  blockfile $in read auto
  close $in
}

setmd time_step 0.001

for { set i 1 } { $i <= 1000 } { incr i } {
  integrate 100
  if {$i % 20 == 0} {
    #puts "."
    repart rand
  } else {
    #puts -nonewline "."
    #flush stdout
  }
  set energies [analyze energy]
  set total [expr [lindex $energies 0 1]/$n_part]
  set kinetic [expr [lindex $energies 1 1]/$n_part]
  set potential [expr [lindex $energies 2 3]/$n_part]
  set pressure [analyze pressure total]
  set kinetic_temperature [expr [lindex $energies 1 1]/(1.5*[setmd n_part])]
  set temperature $kinetic_temperature
  puts "$pressure  $temperature $kinetic  $potential $total"
}

exit
