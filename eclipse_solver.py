import datetime as dat
import numpy as np
import scipy.integrate as sint


# Constants in Astronomical Units

espilon = 1e-13  # Numerical zero

G = 0.01720209895**2  # Gravtionnal constant

# Planet radiuses
r_s = 0.00465047
r_e = 4.25875046E-5
r_m = 1.16138017E-5

# Masses relatively to that of the sun
masses = [1, 1 / 332946.05, 3.694 / 100000000]

# Approximation:
# Bodies are spheres and THAT'S IT


def isEclipse(S, E, M, r_s, r_e, r_m):
    """returns a tuple of boolean, (partial, total)
       S, E, M: np array [x,y,z] of the position of the Sun, the Earth and the Moon
       r_s, r_e, r_m their radius"""
    partial, total = False, False

    earth_sun = S - E
    moon_sun = S - M

    # The moon must be between the earth and the sun
    if norm(moon_sun) > norm(earth_sun):
        return partial, total

    # The point of the moon's sun-facing plane which is on the sun-earth line
    earth_on_moon_plane = calcluateIntersectionPoint(
        M, moon_sun, S, E)

    # Moon to Earth's Projection vector
    M_EP = earth_on_moon_plane - M

    # Normalization
    M_EP = M_EP / norm(M_EP)

    moon_point = M + r_m * M_EP

    sun_point_partial = S - r_s * M_EP
    sun_point_total = S + r_s * M_EP

    # moon_point is the point of the moon that is the closest to the sun-earth line
    # hence the bit of shadow that is the closest to the earth's center is on the line that
    # goes from sun_point_partial (for partial eclipse) to moon_point
    # from sun_point_total (for total eclipse) to moon_point

    shadow_for_partial = calcluateIntersectionPoint(
        E, earth_sun, sun_point_partial, moon_point)

    shadow_for_total = calcluateIntersectionPoint(
        E, earth_sun, sun_point_total, moon_point)

    if distance(shadow_for_partial, E) < r_e:
        partial = True

    if distance(shadow_for_total, E) < r_e:
        total = True

    return partial, total


def norm(V):
    return np.sqrt(sum(V**2))


def squarredDistance(A, B):
    """returns the squarred of the distance AB"""
    return sum((B - A)**2)


def distance(A, B):
    """returns the squarred of the distance AB"""
    return np.sqrt(squarredDistance(A, B))


def calcluateIntersectionPoint(O, n, A, B):
    """retruns, if it exists, the intersection point between:
       -the plane of normal vector n containing O
       -the line (AB)"""
    AB = B - A
    if abs(np.dot(AB, n)) < espilon:
        raise ValueError("No intersection")

    d = -(np.dot(A - O, n) / np.dot(AB, n))

    X = A + d * AB

    return X


def f(y, t):
    """y: flat array of bodies [x1,y1,z1,dx1,dy1,dz1, x2, y2, ...]
       returns a flat array
       to be used in odeint"""
    s = np.zeros((nb_bodies, 6))
    y = y.reshape((nb_bodies, 6))
    for i in range(nb_bodies):
        for j in range(0, nb_bodies):
            if i != j:
                a = y[j] - y[i]
                d = a[0]**2 + a[1]**2 + a[2]**2  # squarred distance
                s[i][3:6] += a[0:3] / d**1.5 * masses[j]
        s[i][3:6] *= G
        s[i][0:3] = y[i][3:6]
    y = y.reshape(nb_bodies * 6)
    return s.reshape(nb_bodies * 6)


# Postions on 01/03/2016 x, y, z, x', y', z' (in atronomical units)

nb_bodies = 3
bodies = np.zeros((nb_bodies, 6))

# Sun
bodies[0] = np.array([0, 0, 0, 0, 0, 0])

# Earth
bodies[1] = np.array([-0.9346772622499, 0.3289072016909, -0.0000103254828,
                      - 0.0059966649168, -0.0162905090794, 0.0000004082766])

# Moon
bodies[2] = np.array([-0.9360196574727, 0.3265999131309, 0.0002193474263,
                      -0.0054985930819, -0.0165531391107, 0.0000177580726])


init_date = dat.datetime(2016, 3, 1)
end_date = dat.datetime(2016, 3, 10)
nb_days = (end_date - init_date).days

t = np.linspace(0, nb_days, 3600 * 24 * nb_days + 1)

# Integartion
sol = sint.odeint(f, bodies.reshape(nb_bodies * 6), t)
sol = sol.reshape((len(t), nb_bodies, 6))

partial_eclipse = [False] * len(t)
total_eclipse = [False] * len(t)

# Time for starting the eclispe search (One day before, 8 march, 22h)
t0 = 3600 * 24 * (nb_days - 1) - 2 * 3600

beg_partial = None
end_partial = None

beg_total = None
end_total = None

# Checking for the eclipse
for i in range(t0, len(t)):
    S, E, M = sol[i, 0, :3], sol[i, 1, :3], sol[i, 2, :3]

    # Is there an eclipse ?
    partial_eclipse[i], total_eclipse[i] = isEclipse(S, E, M, r_s, r_e, r_m)

    # Recording the dates

    # Begin
    if partial_eclipse[i] and not beg_partial:
        beg_partial = (init_date + dat.timedelta(t[i])).isoformat()
    # End
    if not partial_eclipse[i] and beg_partial and not end_partial:
        end_partial = (init_date + dat.timedelta(t[i])).isoformat()

    # Begin
    if total_eclipse[i] and not beg_total:
        beg_total = (init_date + dat.timedelta(t[i])).isoformat()
    # End
    if not total_eclipse[i] and beg_total and not end_total:
        end_total = (init_date + dat.timedelta(t[i])).isoformat()

# Printing the result
print("Partial Eclipse")
print('Beginning: ', beg_partial)
print('End:       ', end_partial)
print("")
print("Total Eclipse")
print('Beginning: ', beg_total)
print('End:       ', end_total)
