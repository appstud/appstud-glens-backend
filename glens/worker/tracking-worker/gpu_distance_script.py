#def calculateDistancePytorch(y,args : List[Tensor]):
def calculateDistancePytorch(y,x):
    #print(args)
    #x=torch.stack(args)
    r1 = torch.sum(x*x,1,keepdim=False)
    r2 = torch.sum(y*y,1,keepdim=False)
    r1 = torch.reshape(r1, [-1, 1])
    r2 = torch.reshape(r2, [1, -1])
    D = r1 - 2*torch.matmul(x, torch.transpose(y,0,1)) + torch.repeat_interleave(r2,repeats=r1.size()[0],dim=0)
    D=torch.sqrt(D)
    return D



