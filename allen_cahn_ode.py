#! /usr/bin/env python3
#
import numpy as np
nx = 101
u = np.zeros ( nx )
x = np.zeros ( nx )

def allen_cahn_deriv ( t, u, x, xi, nu ):

#*****************************************************************************80
#
## allen_cahn_deriv returns the right hand side of the allen-cahn ODE.
#
#  Discussion:
#
#    This version of the equation has the form:
#
#      du/dt = nu * uxx - u * (u^2-1) / (2*xi)
#
#    uxx is the second derivative of u with respect to x
#    nu is a positive diffusion coefficient
#    xi is a positive parameter called the interface thickness.
#
#    Zero Neumann boundary conditions are assumed, and so we
#    set dudt(1) = dudt(2) and dudt(nx) = dudt(nx-1).
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    30 June 2020
#
#  Author:
#
#    John Burkardt
#
#  Reference:
#
#    Jian Zhang, Qiang Du,
#    Numerical studies of discrete approximations to the Allen-Cahn
#    equation in the sharp interface limit,
#    SIAM Journal on Scientific Computing,
#    Volume 31, Number 4, pages 3042-3063, 2009.
#
#  Input:
#
#    real T, the current time.
#
#    real U(*): the current state values.
#
#    real X(*): the node coordinates.
#
#    real XI: the interface thickness.
#
#    real NU: the diffusion coefficient.
#
#  Output:
#
#    real DUDT(*), the time derivatives of the current state values.
#
  import numpy as np

  nx = len ( x )

  uxx = laplacian_interval ( u, x )

  dudt = nu * uxx - u * ( u**2 - 1.0 ) / ( 2.0 * xi**2 )
#
#  Apply boundary conditions.
#
  dudt[0] = dudt[1]
  dudt[nx-1] = dudt[nx-2]

  return dudt

def allen_cahn_initial_condition ( x ):

#*****************************************************************************80
#
## allen_cahn_initial_condition: initial condition for the Allen-Cahn equation.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    30 June 2020
#
#  Author:
#
#    John Burkardt, Catalin Trenchea
#
#  Reference:
#
#    Costica Morosanu, Silvio Paval,
#    On the numerical approximation of a nonlinear reaction-diffusion
#    equation with non-homogeneous Neumann boundary conditions.  Case 1D,
#    Submitted.
#
#  Input:
#
#    real X(*): points at which the initial condition is desired.
#
#  Output:
#
#    real U0(*): the initial condition values.
#
  from scipy import interpolate
  import numpy as np

  nx = len ( x )

  xmin = min ( x )
  xmax = max ( x )

  x0 = np.linspace ( xmin, xmax, 84 )

  v0 = np.array ( [ \
    -1.40, -1.40, -1.44, -1.42, -1.42, \
    -1.44, -1.43, -1.43, -1.42, -1.42, \
    -1.40, -1.40, -1.25, -1.20, -1.17, \
    -1.15, -1.10, -1.08, -1.00, -0.95, \
    -0.90, -0.85, -0.88, -0.60,  0.00, \
     0.50, -0.92, -0.25,  0.80, -0.70, \
     0.58,  0.75,  0.58, -0.63, -0.59, \
     0.69, -0.72,  0.70, -0.59, -0.50, \
     0.70, -0.79, -0.87, -0.88,  0.00, \
     0.72, -0.80,  0.81,  0.00, -0.89, \
     0.00,  0.70,  0.55,  0.68, -0.49, \
     0.79,  0.00, -0.10, -0.80, -0.78, \
    -0.83,  0.69, -0.80,  0.68,  0.50, \
     0.70,  0.59,  1.00,  1.08,  1.10, \
     1.15,  1.17,  1.20,  1.25,  1.30, \
     1.30,  1.25,  1.24,  1.30,  1.31, \
     1.30,  1.32,  1.30,  1.30 ] )
#
#  S=0 means no smoothing is done.
#
  tck = interpolate.splrep ( x0, v0, s = 0 )
#
#  Use a cubic spline to interpolate the data to the points x.
#
  u0 = interpolate.splev ( x, tck, der = 0 )
#
#  Guarantee boundary conditions.
#
  u0[0] = u0[1]
  u0[nx-1] = u0[nx-2]

  return u0

def allen_cahn_ode_euler_display ( xi = 0.015, nu = 1.0 ):

#*****************************************************************************80
#
## allen_cahn_ode_euler_display solves the Allen-Cahn equation using Euler. 
#
#  Discussion:
#
#    This version of the code displays the sequence of solutions to the screen.
#
#    This version of the equation has the form:
#
#      du/dt = nu * uxx - u * (u^2-1)/(2*xi)
#
#    uxx is the second derivative of u with respect to x
#    nu is a positive diffusion coefficient
#    xi is a positive parameter called the interface thickness.
#
#    The variable u is allowed the range -1 <= u <= +1.  The long-term
#    behavior of the solution is to separate into areas with u=+1 and
#    u=-1, with an intermediate interface whose width is related to xi.
#
#    The problem is solved on a 1D interval.
#
#    The boundary conditions on both ends are du/dn = 0.
#    This is handled by setting u(1)=u(2), and u(nx)=u(nx-1).
#
#    The parameter xi is described as an interface thickness.
#
#    For this problem, if nu=1, three humps appear and then:
#      xi = 0.01   all humps stay
#      xi = 0.015  one hump goes away
#      xi = 0.0155 two humps slowly disappear at different times
#      xi = 0.02   two humps disappear fast.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    07 July 2020
#
#  Author:
#
#    John Burkardt, Catalin Trenchea
#
#  Reference:
#
#    Jian Zhang, Qiang Du,
#    Numerical studies of discrete approximations to the Allen-Cahn
#    equation in the sharp interface limit,
#    SIAM Journal on Scientific Computing,
#    Volume 31, Number 4, pages 3042-3063, 2009.
#
#  Input:
#
#    real XI, the interface thickness.  Default is 0.015.
#
#    real NU, the diffusion coefficient.  Default is 1.0.
#    For stability in this code, NU <= 1 is recommended.
#
  import numpy as np
  import matplotlib
  import matplotlib.pyplot as plt
  from matplotlib.animation import FuncAnimation

  print ( '' )
  print ( 'allen_cahn_ode_euler_display' )
  print ( '  Use the Euler method to solve the' )
  print ( '  Allen-Cahn 1D reaction-diffusion equation.' )
  print ( '  Interface thickness xi = %g' % ( xi ) )
  print ( '  Diffusion coefficient nu = %g' % ( nu ) )
  print ( '  Solutions are displayed on the screen immediately.' )

  xmin = 0.0
  xmax = 2.0
  x = np.linspace ( xmin, xmax, nx )
  dx = ( xmax - xmin ) / ( nx - 1 )
#
#  Set the T values.
#
  nt = 1501
  tmin = 0.0
  tmax = 0.15
  t = np.linspace ( tmin, tmax, nt )
  dt = ( tmax - tmin ) / ( nt - 1 )

  u = np.zeros ( nx )
#
#  Discretization report.
#
  print ( '  Use %d nodes, with dx=%g in space interval [%g,%g]' \
    % ( nx, dx, xmin, xmax ) )
  print ( '  Use %d nodes, with dt=%g in time interval [%g,%g]' \
    % ( nt, dt, tmin, tmax ) )
#
#  Set up the animation frame.
#
  fig, ax = plt.subplots ( )
  ax.set_xlim ( xmin, xmax )
  ax.set_ylim ( -1.1, +1.1 )
  plt.grid ( True )
  plt.title ( 'Allen-Cahn equation xi = %g, nu =%g' % ( xi, nu ) )
  line, = ax.plot ( 0, 0, linewidth = 2 )

  def animation_init():
    line.set_data ( x, u )
    return line,
#
#  Moronic MATPLOTLIB animation needs this hideous "global u" statement!
#  Inconsistent MATPLOTLIB does NOT need a "global x" statement.
#
  def animation_frame ( j ):

    global u

    if ( j == 0 ):
      u = allen_cahn_initial_condition ( x )
    else:
      dudt = allen_cahn_deriv ( t, u, x, xi, nu )
      u = u + dudt * dt

    line.set_xdata ( x )
    line.set_ydata ( u )
    return line, 

  animation = FuncAnimation ( fig, init_func = animation_init, \
    func = animation_frame, 
    frames = np.arange ( 0, nt ), interval = 10, repeat = False )
#
#  To actually see the plots interactively, set block = True.
#
  plt.show ( block = False )
#
#  Terminate.
#
  print ( '' )
  print ( 'allen_cahn_ode_euler_display' )
  print ( '  Normal end of execution.' )

  return

def allen_cahn_ode_euler_movie ( xi = 0.015, nu = 1.0 ):

#*****************************************************************************80
#
## allen_cahn_ode_euler_movie solves the Allen-Cahn equation using Euler. 
#
#  Discussion:
#
#    This version of the code creates a movie of the solution sequence.
#
#    This version of the equation has the form:
#
#      du/dt = nu * uxx - u * (u^2-1)/(2*xi)
#
#    uxx is the second derivative of u with respect to x
#    nu is a positive diffusion coefficient
#    xi is a positive parameter called the interface thickness.
#
#    The variable u is allowed the range -1 <= u <= +1.  The long-term
#    behavior of the solution is to separate into areas with u=+1 and
#    u=-1, with an intermediate interface whose width is related to xi.
#
#    The problem is solved on a 1D interval.
#
#    The boundary conditions on both ends are du/dn = 0.
#    This is handled by setting u(1)=u(2), and u(nx)=u(nx-1).
#
#    The parameter xi is described as an interface thickness.
#
#    For this problem, if nu=1, three humps appear and then:
#      xi = 0.01   all humps stay
#      xi = 0.015  one hump goes away
#      xi = 0.0155 two humps slowly disappear at different times
#      xi = 0.02   two humps disappear fast.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    07 July 2020
#
#  Author:
#
#    John Burkardt, Catalin Trenchea
#
#  Reference:
#
#    Jian Zhang, Qiang Du,
#    Numerical studies of discrete approximations to the Allen-Cahn
#    equation in the sharp interface limit,
#    SIAM Journal on Scientific Computing,
#    Volume 31, Number 4, pages 3042-3063, 2009.
#
#  Input:
#
#    real XI, the interface thickness.  Default is 0.015.
#
#    real NU, the diffusion coefficient.  Default is 1.0.
#    For stability in this code, NU <= 1 is recommended.
#
  import numpy as np
  import matplotlib
  import matplotlib.pyplot as plt
  from matplotlib.animation import FuncAnimation

  matplotlib.use ( 'Agg' )

  print ( '' )
  print ( 'allen_cahn_ode_euler_movie' )
  print ( '  Use the Euler method to solve the' )
  print ( '  Allen-Cahn 1D reaction-diffusion equation.' )
  print ( '  Interface thickness xi = %g' % ( xi ) )
  print ( '  Diffusion coefficient nu = %g' % ( nu ) )
  print ( '  The solutions are written to a movie file.' )

  xmin = 0.0
  xmax = 2.0
  x = np.linspace ( xmin, xmax, nx )
  dx = ( xmax - xmin ) / ( nx - 1 )
#
#  Set the T values.
#
  nt = 1501
  tmin = 0.0
  tmax = 0.15
  t = np.linspace ( tmin, tmax, nt )
  dt = ( tmax - tmin ) / ( nt - 1 )
#
#  Discretization report.
#
  print ( '  Use %d nodes, with dx=%g in space interval [%g,%g]' \
    % ( nx, dx, xmin, xmax ) )
  print ( '  Use %d nodes, with dt=%g in time interval [%g,%g]' \
    % ( nt, dt, tmin, tmax ) )
#
#  Set up the animation frame.
#
  fig, ax = plt.subplots ( )
  ax.set_xlim ( xmin, xmax )
  ax.set_ylim ( -1.1, +1.1 )
  plt.grid ( True )
  plt.title ( 'Allen-Cahn equation xi = %g, nu =%g' % ( xi, nu ) )
  line, = ax.plot ( 0, 0, linewidth = 2 )

  def animation_init():
    line.set_data ( x, u )
    return line,

  def animation_frame ( j ):

    global u

    if ( j == 0 ):
      u = allen_cahn_initial_condition ( x )
    else:
      dudt = allen_cahn_deriv ( t, u, x, xi, nu )
      u = u + dudt * dt

    line.set_xdata ( x )
    line.set_ydata ( u )
    return line, 

  animation = FuncAnimation ( fig, init_func = animation_init, \
    func = animation_frame, \
    frames = np.arange ( 0, nt ), \
    interval = 200, \
    repeat = False )

  filename = 'allen_cahn_ode_euler_movie.mp4'

  animation.save ( filename, fps = 20 )

  print ( '  Graphics saved as "%s"' % ( filename ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'allen_cahn_ode_euler_movie' )
  print ( '  Normal end of execution.' )

  return

def allen_cahn_ode_test ( ):

#*****************************************************************************80
#
## allen_cahn_ode_test solves the Allen-Cahn equation.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    07 July 2020
#
#  Author:
#
#    John Burkardt, Catalin Trenchea
#
  import platform

  print ( '' )
  print ( 'allen_cahn_ode_test' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  Use ODE solvers on the Allen-Cahn equation.' )

  allen_cahn_ode_euler_display ( )
  allen_cahn_ode_euler_movie ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'allen_cahn_ode_test' )
  print ( '  Normal end of execution.' )

  return

def laplacian_interval ( u, x ):

#*****************************************************************************80
#
## laplacian_interval approximates the laplacian on an interval.
#
#  Discussion:
#
#    The domain is represented by nx equally spaced points.
#
#    The laplacian is computed for the interior points, but a value
#    of 0 is returned for the first and last points.
#
#    It is up to the user to decide whether to reset l(1) and l(nx)
#    to nonzero values.
#
#  Modified:
#
#    30 June 2020
#
#  Input:
#
#    real u(nx): the values of a function on an equally spaced grid of points.
#
#    real x(nx): the node coordinates.
#
#  Output:
#
#    real uxx(nx): the estimate for the laplacian.
#
  import numpy as np

  nx = len ( u )

  uxl = ( u[1:nx-1] - u[0:nx-2] ) / ( x[1:nx-1] - x[0:nx-2] )
  uxr = ( u[2:nx]   - u[1:nx-1] ) / ( x[2:nx]   - x[1:nx-1] )

  uxx = 2.0 * ( uxr[0:nx-2] - uxl[0:nx-2] ) / ( x[2:nx] - x[0:nx-2] )
#
#  Pad the result with initial and final 0.
#
  uxx = np.insert ( uxx, 0, 0.0 )
  uxx = np.insert ( uxx, nx-1, 0.0 )

  return uxx

def timestamp ( ):

#*****************************************************************************80
#
## timestamp prints the date as a timestamp.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    21 August 2019
#
#  Author:
#
#    John Burkardt
#
  import time

  t = time.time ( )
  print ( time.ctime ( t ) )

  return

if ( __name__ == '__main__' ):
  timestamp ( )
  allen_cahn_ode_test ( )
  timestamp ( )
