module SpatialCoherence

include("MatrixTrans.jl")

using Base.Threads

export memberPC, memberPCmt, ensemblePC, ensemblePCmt, memberPCVB, ensemblePCVB

"""
     memberPC(Useed, Xs, Ys, w0, TopologicalCharge, RadialOrder, circle, N)

Simulates a member of the ensamble to generate a partially coherent Useed beam """
function memberPC(Useed, ren, col, XYsize, circle::Float64, N::Int64)

# Preallocates memory for matrices U and Uaux
Uaux = zeros(ComplexF64, ren, col)
U = zeros(ComplexF64, ren, col)

# Uniformly distributed vortices in a circle of radius "circle"
r = circle*sqrt.(rand(N,1))
theta = 2*pi*rand(N,1)
phi = 2*pi*rand(N,1)
xv = Integer.(div.(r.*cos.(theta),XYsize))
yv = Integer.(div.(r.*sin.(theta),XYsize))

# Coherent superposition
for member=1:N
    Uaux = exp(im * phi[member]) .* MatrixTranslate(Useed, xv[member], yv[member])
    Uaux = Uaux./maximum(abs.(Uaux))
    U = Uaux + U
end

return U
end

"""
     memberPCmt(Useed, Xs, Ys, w0, TopologicalCharge, RadialOrder, circle, N)

Simulates a member of the ensamble to generate a partially coherent Useed beam (multithreading)"""
function memberPCmt(Useed, ren, col, XYsize, circle::Float64, N::Int64)

# Preallocates memory for matrices U and Uaux
Uaux = zeros(ComplexF64, ren, col)
U = zeros(ComplexF64, ren, col)

# Uniformly distributed vortices in a circle of radius "circle"
r = circle*sqrt.(rand(N,1))
theta = 2*pi*rand(N,1)
phi = 2*pi*rand(N,1)
xv = Integer.(div.(r.*cos.(theta),XYsize))
yv = Integer.(div.(r.*sin.(theta),XYsize))

# Coherent superposition
@Threads.threads for member=1:N
    Uaux = exp(im * phi[member]) .* MatrixTranslate(Useed, xv[member], yv[member])
    Uaux = Uaux./maximum(abs.(Uaux))
    U = Uaux + U
end

return U
end


function ensemblePC(Useed, Vsize, Hsize, XYsize, circle, N, Ne)

Isum = zeros(Float64, Vsize, Hsize)
Xisum = zeros(Float64, Vsize, Hsize)

    # Generates the desired holograms according to the given parameters
    for memberE = 1:Ne

        # Computes the field of one member
        Upc = memberPC(Useed, Vsize, Hsize, XYsize, circle, N)

        # Incoherent superposition
        I = abs2.(Upc)
        Xi=real(Upc .* conj(rot180(Upc)))
        Isum=Isum+I
        Xisum=Xisum+Xi
    end

    # Averaging
    Iaverage=Isum/Ne
    Xiaverage=Xisum/Ne

return Iaverage::Array{Float64, 2}, Xiaverage::Array{Float64, 2}

end


function ensemblePCmt(Useed, Vsize, Hsize, XYsize, circle, N, Ne)

Isum = zeros(Float64, Vsize, Hsize)
Xisum = zeros(Float64, Vsize, Hsize)

    # Generates the desired holograms according to the given parameters
    for memberE = 1:Ne

        # Computes the field of one member
        Upc = memberPCmt(Useed, Vsize, Hsize, XYsize, circle, N)

        # Incoherent superposition
        I = abs2.(Upc)
        Xi = real(Upc .* conj(rot180(Upc)))
        Isum = Isum+I
        Xisum = Xisum+Xi
    end

    # Averaging
    Iaverage = Isum/Ne
    Xiaverage = Xisum/Ne

return Iaverage::Array{Float64, 2}, Xiaverage::Array{Float64, 2}

end



"""
     memberPCVB(UseedR, UseedL, Xs, Ys, w0, TopologicalCharge, RadialOrder, circle, N)

Simulates a member of the ensamble to generate a partially coherent Useed beam """
function memberPCVB(UseedR, UseedL, ren, col, XYsize, circle::Float64, N::Int64)

# Preallocates memory for matrices U and Uaux
UauxR = zeros(ComplexF64, ren, col)
UR = zeros(ComplexF64, ren, col)
UauxL = zeros(ComplexF64, ren, col)
UL = zeros(ComplexF64, ren, col)

# Uniformly distributed vortices in a circle of radius "circle"
r = circle*sqrt.(rand(N,1))
theta = 2*pi*rand(N,1)
phi = 2*pi*rand(N,1)
xv = Integer.(div.(r.*cos.(theta),XYsize))
yv = Integer.(div.(r.*sin.(theta),XYsize))

# Coherent superposition
for member=1:N
    # Right component
    UauxR = exp(im * phi[member]) .* MatrixTranslate(UseedR, xv[member], yv[member])
    # UauxR = UauxR./maximum(abs.(UauxR))
    UauxR = UauxR./N
    UR = UauxR + UR

    # Left component
    UauxL = exp(im * phi[member]) .* MatrixTranslate(UseedL, xv[member], yv[member])
    # UauxL = UauxL./maximum(abs.(UauxL))
    UauxL = UauxL./N
    UL = UauxL + UL
end

return UR, UL
end

function ensemblePCVB(UseedR, UseedL, Vsize, Hsize, XYsize, circle, N, Ne)

Isum = zeros(Float64, Vsize, Hsize)
Xisum = zeros(Float64, Vsize, Hsize)
IsumX = zeros(Float64, Vsize, Hsize)
XisumX = zeros(Float64, Vsize, Hsize)
IsumY = zeros(Float64, Vsize, Hsize)
XisumY = zeros(Float64, Vsize, Hsize)
IsumD = zeros(Float64, Vsize, Hsize)
XisumD = zeros(Float64, Vsize, Hsize)
IsumA = zeros(Float64, Vsize, Hsize)
XisumA = zeros(Float64, Vsize, Hsize)
IsumR = zeros(Float64, Vsize, Hsize)
XisumR = zeros(Float64, Vsize, Hsize)
IsumL = zeros(Float64, Vsize, Hsize)
XisumL = zeros(Float64, Vsize, Hsize)

    # Generates the desired holograms according to the given parameters
    for memberE = 1:Ne

        # Computes the field of one member
        UpcR, UpcL = memberPCVB(UseedR, UseedL, Vsize, Hsize, XYsize, circle, N)

        # Convert to EX, EY
        UpcX = (sqrt(2)/2) .* (UpcL .+ UpcR)
        UpcY = (sqrt(2)/(2*im)) .* (UpcL .- UpcR)

        # Compute the Stokes measurements (just diagonal and antidiagonal)
        UpcD = (sqrt(2)/2) .* (UpcX .+ UpcY)
        UpcA = (sqrt(2)/2) .* (-UpcX .+ UpcY)

        # Incoherent superposition
        I = abs2.(UpcL) .+ abs2.(UpcR)
        I = abs2.(I)
        Xi=real(UpcR .* conj(rot180(UpcR))) + real(UpcL .* conj(rot180(UpcL)))
        Isum=Isum+I
        Xisum=Xisum+Xi

        # Incoherent superposition
        IX = abs2.(UpcX)
        XiX=real(UpcX .* conj(rot180(UpcX)))
        IsumX=IsumX+IX
        XisumX=XisumX+XiX

        IY = abs2.(UpcY)
        XiY=real(UpcY .* conj(rot180(UpcY)))
        IsumY=IsumY+IY
        XisumY=XisumY+XiY

        ID = abs2.(UpcD)
        XiD=real(UpcD .* conj(rot180(UpcD)))
        IsumD=IsumD+ID
        XisumD=XisumD+XiD

        IA = abs2.(UpcA)
        XiA=real(UpcA .* conj(rot180(UpcA)))
        IsumA=IsumA+IA
        XisumA=XisumA+XiA

        IR = abs2.(UpcR)
        XiR=real(UpcR .* conj(rot180(UpcR)))
        IsumR=IsumR+IR
        XisumR=XisumR+XiR

        IL = abs2.(UpcL)
        XiL=real(UpcL .* conj(rot180(UpcL)))
        IsumL=IsumL+IL
        XisumL=XisumL+XiL

    end

    # Averaging
    Iaverage=Isum/Ne
    Xiaverage=Xisum/Ne

    IaverageX=IsumX/Ne
    XiaverageX=XisumX/Ne

    IaverageY=IsumY/Ne
    XiaverageY=XisumY/Ne

    IaverageD=IsumD/Ne
    XiaverageD=XisumD/Ne

    IaverageA=IsumA/Ne
    XiaverageA=XisumA/Ne

    IaverageR=IsumR/Ne
    XiaverageR=XisumR/Ne

    IaverageL=IsumL/Ne
    XiaverageL=XisumL/Ne

# Return all measurements!  (might be slow :C)
return Iaverage, Xiaverage, IaverageX, XiaverageX, IaverageY, XiaverageY, IaverageD, XiaverageD, IaverageA, XiaverageA, IaverageR, XiaverageR, IaverageL, XiaverageL

end

end # module
