import torch


def wrap_query_label(q_label1, r_label1, q_label2, r_label2, mode):
    assert q_label1.size(1) == r_label1.size(1), 'lenght error!'
    assert q_label2.size(1) == r_label2.size(1), 'lenght error!'
    
    # creat end and start labels
    r1, r2 = q_label1.size(0), q_label2.size(0)
    c1, c2 = q_label1.size(1), q_label2.size(1)
    print(r1, r2)
    print(c1, c2)
    
    l1 = torch.zeros(r1, c2)
    l2 = torch.zeros(r2, c1)
    
    # creat new query labels
    if mode == 'all':
        q1 = torch.cat([q_label1, l1], dim=1)
        q2 = torch.cat([l2, q_label2], dim=1)
        assert q1.size(1) == q2.size(1), 'cat error!'
        
        Q = torch.cat([q1, q2], dim=0)
        print(Q.size(), mode)
        
    elif mode == 'd1':
        q1 = torch.cat([q_label1, l1], dim=1)   
        Q = q1
        print(Q.size(), mode)
        
    elif mode == 'd2':
        q2 = torch.cat([l2, q_label2], dim=1)    
        Q = q2
        print(Q.size(), mode)
        
    else:
        raise ValueError('mode error!')
        
    return Q
    
    
def wrap_retrieval_label(r_label1, r_label2):
    # creat end and start labels
    r1, r2 = r_label1.size(0), r_label2.size(0)
    c1, c2 = r_label1.size(1), r_label2.size(1)
    print(r1, r2)
    print(c1, c2)
    
    l1 = torch.zeros(r1, c2)
    l2 = torch.zeros(r2, c1)
    
    
    # creat new retrieval labels
    r1 = torch.cat([r_label1, l1], dim=1)
    r2 = torch.cat([l2, r_label2], dim=1)
    
    assert r1.size(1) == r2.size(1), 'cat error!'
    
    R = torch.cat([r1, r2], dim=0)
    print(R.size())
    
    return R


# a1 = torch.tensor([[1,1,1,1,1],[1,1,1,1,1]]).float()
# a2 = torch.tensor([[1,1,1,0,0,0],[1,1,1,0,0,0]]).float()
# b1 = torch.tensor([[0,0,0,1,1],[0,0,0,1,1]]).float()
# b2 = torch.tensor([[0,0,0,1,1,1],[0,0,0,1,1,1]]).float()


# c,d = wrap_label(a1,b1,a2,b2)
# print(c)
# print(d)
    
    

