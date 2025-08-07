import pickle
from models.Model import RWNet
from utils import *
import pandas as pd
import numpy as np
from datetime import datetime
from torch_geometric.loader import DataLoader
from lifelines.utils import concordance_index
import argparse
import torch

def test(model, device, loader, ppi_adj, new_ppi_features):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            mol_data = data[0].to(device)
            pro_data = data[1].to(device)
            output = model(mol_data, pro_data, ppi_adj, new_ppi_features)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, mol_data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def main(args):
    dataset = args.dataset
    model_dict_ = {'RWNet': RWNet}
    modeling = model_dict_[args.model]
    model_st = modeling.__name__

    path = f'results/{dataset}/train_{model_st}_best.model'
    # device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")
    device = torch.device(f'cuda:{args.device}')

    check_point = torch.load(path, map_location=device)
    model = modeling()
    model.load_state_dict(check_point)

    model = model.to(device)
    df_test = pd.read_csv(f'data/{dataset}/test.csv')
    test_smile, test_seq, test_label = list(df_test['compound_iso_smiles']), list(df_test['target_sequence']), list(df_test['affinity'])

    with open(f"data/{dataset}/mol_data.pkl", 'rb') as file:
        mol_data = pickle.load(file)
    with open(f'data/{dataset}/pro_data.pkl', 'rb') as file2:
        pro_data = pickle.load(file2)
    with open(f'data/{dataset}/PPI/ppi_data_wv.pkl', 'rb') as file3:
        ppi_adj, new_ppi_features, ppi_index = pickle.load(file3)
    #with open(f'data/{dataset}/PPI/ppi_data.pkl', 'rb') as file3:
    #    ppi_adj, ppi_features, ppi_index = pickle.load(file3)

    ppi_adj = torch.LongTensor(np.argwhere(ppi_adj == 1).transpose(1, 0)).to(device)
    new_ppi_features = torch.Tensor(new_ppi_features).to(device)

    pro_graph = proGraph(pro_data, ppi_index, device)
    test_dataset = DTADataset(test_smile, test_seq, test_label, mol_data=mol_data, ppi_index=ppi_index, new_ppi_features = new_ppi_features)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    G, P = test(model, device, test_loader, ppi_adj, new_ppi_features)
    rm2_score = get_rm2(G, P)
    ret = [mse(G, P), concordance_index(G, P), rm2_score]
    # G_list = G.tolist()
    # P_list = P.tolist()
    # predicted_data = {
    #     'smile':test_smile,
    #     'sequence':test_seq,
    #     'label':G_list,
    #     'predicted value':P_list
    # }
    # df_pre = pd.DataFrame(predicted_data)
    # df_pre.to_csv(f'./results/{args.dataset}/predicted_value_of_{args.model}_on_{args.dataset} .csv')
    # ret = [mse(G, P),concordance_index(G, P)]
    print(args.dataset, model_st, 'test_mse:', ret[0], 'test_ci:', ret[1], 'test_rm2', ret[2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'RWNet', choices = ['RWNet'])
    parser.add_argument('--batch', type = int, default = 512)
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--dataset', type = str, default = 'kiba',choices=['davis', 'kiba'])
    parser.add_argument('--num_workers', type= int, default = 0)
    args = parser.parse_args()
    beginT = datetime.now()
    print("Starting Time: {}".format(beginT.strftime(r'%m-%d-%H:%M:%S')))
    main(args)
    endT = datetime.now()
    print("Ending Time: {}".format(endT.strftime(r'%m-%d-%H:%M:%S')))
    interval = endT - beginT
    m, s = divmod(interval.total_seconds(), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    print("Duration is {}d {}h {}m {}s".format(d, h, m, s))
