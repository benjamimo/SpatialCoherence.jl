"""
    MatrixTranslate(U, xv::Int64, yv::Int64)

Dirty matrix translation """
function MatrixTranslate(U, xv::Int64, yv::Int64)

# Preallocates memory for matrices U and Uaux
ren = size(U,1)
col = size(U,2)
V = zeros(ComplexF64, ren, col)

# Shifts!
if xv>=0 && yv>=0
    V[yv+1:end, xv+1:end] = U[1:end-yv, 1:end-xv]

elseif xv<=0 && yv>=0
    V[yv+1:end, 1:end-abs(xv)] = U[1:end-yv, abs(xv)+1:end]

elseif xv>=0 && yv<=0
    V[1:end-abs(yv), xv+1:end] = U[abs(yv)+1:end, 1:end-xv]

elseif xv<=0 && yv<=0
    V[1:end-abs(yv), 1:end-abs(xv)] = U[abs(yv)+1:end, abs(xv)+1:end]
end

return V

end

# Dirty solution!
function MatrixUnwrap(p)
    pointint = size(p,1)

    center = div(pointint,2)+1
    unwrp = zeros(pointint, div(pointint,2))

    th = collect(range(0,length=pointint,stop=2*pi))
    for ang = 1:pointint
        aux = imrotate(p,th[ang], axes(p))
        unwrp[ang,:] = aux[center, center:end]
    end
    return unwrp
end
