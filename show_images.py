import torch
from torchvision.utils import make_grid
from torchvision import transforms




def show_topk_images(query_dataloader, retrieval_dataloader, query_code, retrieval_code, topk=99, img_num=3, step='1'):
    '''
    
    query_dataloader: Dataloader
    retrieval_dataloader: Dataloader
    query_code: tensor
    retrieval_code: tensor
    topk: int
    img_num: int
    
    '''

    
    for i in range(img_num):
        
        if i == 0:
            print(torch.all(torch.eq(query_code[0],query_code[1])))
            print(torch.all(torch.eq(query_code[0],query_code[2])))
            print(torch.all(torch.eq(query_code[0],query_code[3])))
            print(torch.all(torch.eq(query_code[0],query_code[4])))
            print(torch.all(torch.eq(query_code[0],query_code[5])))
            print(torch.all(torch.eq(query_code[0],query_code[6])))
            print(torch.all(torch.eq(query_code[0],query_code[7])))
            print(torch.all(torch.eq(query_code[0],query_code[8])))
            print(torch.all(torch.eq(query_code[0],query_code[9])))
            print(len(query_code))
            print(len(retrieval_code))
        distance = query_code.shape[1] - query_code[i] @ retrieval_code.t()
        index = torch.argsort(distance)[:topk]
        print(i, distance[index])
        #print(retrieval_code[index[0]])
        
        query_data = query_dataloader.dataset.__getitem__(i, mode='original')[0]
        retrieval_data = torch.tensor([])
        for y in index:
            retrieval_data = torch.cat([retrieval_data, retrieval_dataloader.dataset.__getitem__(y, mode='original')[0].unsqueeze(0)], dim=0)
            
        data = torch.cat([query_data.unsqueeze(0), retrieval_data], dim=0)
        
        show_images = make_grid(data, nrow=10)
        show_images = transforms.ToPILImage()(show_images)
        show_images = transforms.Resize(640)(show_images)
        show_images.save('./step%s_images/image_show_%d.jpg'%(step, i))
        
        
    print('finish make topk image')
        
   

