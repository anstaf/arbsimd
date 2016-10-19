module PassiveCable

export cable_normalize, cable, rallpack1

# Compute solution g(x, t) to
#
#     ∂²g/∂x² - g - ∂g/∂t = 0
#
# on [0, L] × [0,∞), subject to:
#
#     g(x, 0) = 0
#     ∂g/∂x (0, t) = 1
#     ∂g/∂x (L, t) = 0
#
# (This implementation converges slowly for small t)

function cable_normalized(x, t, L; tol=1e-8)
    if t<=0
        return 0.0
    else
        ginf = -cosh(L-x)/sinh(L)
        sum = exp(-t/L)

        for k = countfrom(1)
            a = k*pi/L
            e = exp(-t*(1+a^2))

            sum += 2/L*e*cos(a*x)/(1+a^2)
            resid_ub = e/(L*a^3*t)

            if resid_ub<tol
                break
            end
        end
	return ginf+sum
     end
end


# Compute solution f(x, t) to
#
#     λ²∂²f/∂x² - f - τ∂f/∂t = 0
#
# on [0, L] x [0, ∞), subject to:
#
#     f(x, 0) = V
#     ∂f/∂x (0, t) = I·r
#     ∂f/∂x (L, t) = 0
#
# where:
#
#     λ² = 1/(r·g)   length constant
#     τ  = r·c       time constant
#
# In the physical model, the parameters correspond to the following:
#
#     L:  length of cable
#     r:  linear axial resistivity
#     g:  linear membrane conductance
#     c:  linear membrane capacitance.
#     V:  membrane reversal potential
#     I:  injected current on the left end (x = 0) of the cable.

function cable(x, t, L, lambda, tau, r, V, I; tol=1e-8)
    scale = I*r*lambda;
    if scale == 0
	return V
    else
        tol_n = abs(tol/scale)
	return scale*cable_normalized(x/lambda, t/tau, L/lambda, tol=tol_n) + V
    end
end


# Rallpack 1 test
#
# One sided cable model with the following parameters:
#
#     RA = 1 Ω·m    bulk axial resistivity
#     RM = 4 Ω·m²   areal membrane resistivity
#     CM = 0.01 F/m²  areal membrane capacitance
#     d  = 1 µm     cable diameter
#     EM = -65 mV   reversal potential
#     I  = 0.1 nA   injected current
#     L  = 1 mm     cable length 
#

function rallpack1(x, t; tol=1e-8)
    RA = 1
    RM = 4
    CM = 1e-2
    d  = 1e-6
    EM = -65e-3
    I  = 0.1e-9
    L  = 1e-3

    r = 4*RA/(pi*d*d)
    lambda = sqrt(d/4 * RM/RA)
    tau = CM*RM

    return cable(x, t, L, lambda, tau, r, EM, -I, tol=tol)
end

end #module
