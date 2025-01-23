

from .model import *
from .utils import *
from .downprompt import featureprompt
from .TxData import TxData


import sys
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)
#device = torch.device("cuda:0")

class TxGNNPrompt:
    
    def __init__(self, data,
                       weight_bias_track = False,
                       proj_name = 'TxGNNPrompt',
                       exp_name = 'TxGNNPrompt',
                       device = 'cuda:0'):
        self.device = torch.device(device)
        self.weight_bias_track = weight_bias_track
        self.G = data.G
        self.df, self.df_train, self.df_valid, self.df_test = data.df, data.df_train, data.df_valid, data.df_test
        self.data_folder = data.data_folder
        self.disease_eval_idx = data.disease_eval_idx
        self.split = data.split
        self.no_kg = data.no_kg
        
        self.disease_rel_types = ['rev_contraindication', 'rev_indication', 'rev_off-label use']
        
        self.dd_etypes= [('drug', 'contraindication', 'disease'),
                  ('drug', 'indication', 'disease'),
                  ('drug', 'off-label use', 'disease'),
                  ('disease', 'rev_contraindication', 'drug'),
                  ('disease', 'rev_indication', 'drug'),
                  ('disease', 'rev_off-label use', 'drug')]
        # self.dd_etypes = [('drug', 'contraindication', 'disease'),
        #                   ('drug', 'indication', 'disease'),
        #                   ('drug', 'off-label use', 'disease'),
        #                   ('disease', 'rev_contraindication', 'drug'),
        #                   ('disease', 'rev_indication', 'drug'),
        #                   ('disease', 'rev_off-label use', 'drug'),
        #                   ('drug', 'drug_drug', 'drug'),
        #                   ('gene/protein', 'protein_protein', 'gene/protein'),
        #                   ('disease', 'disease_disease', 'disease'),
        #                   ('drug', 'drug_protein', 'gene/protein'),
        #           ('gene/protein', 'disease_protein', 'disease')]

        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)
            self.wandb = wandb
        else:
            self.wandb = None
        self.config = None

    def model_initialize(self, n_hid = 128,
                               n_inp = 128,
                               n_out = 128,
                               proto = True,
                               proto_num = 5,
                               sim_measure = 'all_nodes_profile',
                               bert_measure = 'disease_name',
                               agg_measure = 'rarity',
                               exp_lambda = 0.7,
                               num_walks = 200,
                               walk_mode = 'bit',
                               path_length = 2):

        if self.no_kg and proto:
            print('Ablation study on No-KG. No proto learning is used...')
            proto = False

        self.G = self.G.to('cpu')
        self.G = initialize_node_embedding(self.G, n_inp)
        self.g_valid_pos, self.g_valid_neg = evaluate_graph_construct(self.df_valid, self.G, 'fix_dst', 1, self.device)
        self.g_test_pos, self.g_test_neg = evaluate_graph_construct(self.df_test, self.G, 'fix_dst', 1, self.device)

        self.config = {'n_hid': n_hid,
                       'n_inp': n_inp,
                       'n_out': n_out,
                       'proto': proto,
                       'proto_num': proto_num,
                       'sim_measure': sim_measure,
                       'bert_measure': bert_measure,
                       'agg_measure': agg_measure,
                       'num_walks': num_walks,
                       'walk_mode': walk_mode,
                       'path_length': path_length
                      }

        self.model = HeteroRGCN(self.G,
                   in_size=n_inp,
                   hidden_size=n_hid,
                   out_size=n_out,
                   proto = proto,
                   proto_num = proto_num,
                   sim_measure = sim_measure,
                   bert_measure = bert_measure,
                   agg_measure = agg_measure,
                   num_walks = num_walks,
                   walk_mode = walk_mode,
                   path_length = path_length,
                   split = self.split,
                   data_folder = self.data_folder,
                   exp_lambda = exp_lambda,
                   device = self.device
                  ).to(self.device)

    def pretrain(self, n_epoch=1, learning_rate=1e-3, batch_size=1024, train_print_per_n=20, save_model_path='./',
                 sweep_wandb=None):

        if self.no_kg:
            raise ValueError('During No-KG ablation, pretraining is infeasible because it is the same as finetuning...')

        self.G = self.G.to('cpu')
        print('Creating minibatch pretraining dataloader...')
        train_eid_dict = {etype: self.G.edges(form='eid', etype=etype) for etype in self.G.canonical_etypes}

        rel_unique = self.df.relation.unique()
        reverse_etypes = {}
        for rel in rel_unique:
            if 'rev_' in rel:
                reverse_etypes[rel] = rel[4:]
            elif 'rev_' + rel in rel_unique:
                reverse_etypes[rel] = 'rev_' + rel
            else:
                reverse_etypes[rel] = rel

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

        dataloader = dgl.dataloading.DistEdgeDataLoader(
            self.G, train_eid_dict, sampler,
            negative_sampler=Minibatch_NegSampler(self.G, 1, 'fix_dst'),
            shuffle=True,
            batch_size=batch_size,
            drop_last=False,
            num_workers=0)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        print('Start pre-training with #param: %d' % (get_n_params(self.model)))
        self.G = self.G.to(self.device)
        best_auprc = -1
        best_models = []
        best_auprc_list = []
        for epoch in range(n_epoch):

            for step, (nodes, pos_g, neg_g, blocks) in enumerate(dataloader):

                blocks = [i.to(self.device) for i in blocks]
                pos_g = pos_g.to(self.device)
                neg_g = neg_g.to(self.device)
                pred_score_pos, pred_score_neg, pos_score, neg_score, dgi_loss = self.model.forward_minibatch(pos_g,
                                                                                                              neg_g,
                                                                                                              blocks,
                                                                                                              self.G,
                                                                                                              mode='train',
                                                                                                              pretrain_mode=True)

                scores = torch.cat((pos_score, neg_score)).reshape(-1, )
                labels = [1] * len(pos_score) + [0] * len(neg_score)

                loss = F.binary_cross_entropy(scores, torch.Tensor(labels).float().to(self.device))

                all_loss = 0.1 * dgi_loss + loss

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                if self.weight_bias_track:
                    self.wandb.log({"Pretraining Loss": loss})
                if (step) % train_print_per_n == 0:

                    tauroc_rel, tauprc_rel, tmicro_auroc, tmicro_auprc, tmacro_auroc, tmacro_auprc = get_all_metrics_fb(
                        pred_score_pos, pred_score_neg, scores.reshape(-1, ).detach().cpu().numpy(), labels, self.G,
                        True)
                    with torch.no_grad():
                        (auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc,
                         macro_auprc), loss, pred_pos, pred_neg = evaluate_fb(self.model, self.g_test_pos,
                                                                              self.g_test_neg, self.G, self.dd_etypes,
                                                                              self.device, True, mode='test')
                    # Open a text file in write mode
                    with open(save_model_path + "/output.txt", "a") as f:
                        # Redirect stdout to the file
                        sys.stdout = f
                        print(
                            'Epoch: %d Step: %d LR: %.5f Loss %.4f, Pretrain Micro AUROC %.4f Pretrain Micro AUPRC %.4f Pretrain Macro AUROC %.4f Pretrain Macro AUPRC %.4f' % (
                                epoch,
                                step,
                                optimizer.param_groups[0]['lr'],
                                loss,
                                tmicro_auroc,
                                tmicro_auprc,
                                tmacro_auroc,
                                tmacro_auprc
                            ))
                        # Now the following print statements will go to the file instead of the console
                        print(
                            'Testing Loss %.4f Testing Micro AUROC %.4f Testing Micro AUPRC %.4f Testing Macro AUROC %.4f Testing Macro AUPRC %.4f' % (
                                loss,
                                micro_auroc,
                                micro_auprc,
                                macro_auroc,
                                macro_auprc
                            ))
                        print('----- AUROC Performance in Each Relation -----')
                        print(auroc_rel)
                        print('----- AUPRC Performance in Each Relation -----')
                        print(auprc_rel)


                    # Reset stdout back to the console after the block
                    sys.stdout = sys.__stdout__
                    if macro_auprc > best_auprc:
                        best_auprc = macro_auprc
                        checkpoint_path = os.path.join(save_model_path, f'model{epoch}_{step}.pth')
                        torch.save(self.model.state_dict(), checkpoint_path)
                        best_models.append(os.path.join(save_model_path, f'model{epoch}_{step}.pth'))
                        best_auprc_list.append(macro_auprc)
                        if len(best_models) > 5:
                            model_to_delete = best_models.pop(0)
                            os.remove(model_to_delete)


    def prompt(self, n_epoch = 500,
                       learning_rate = 1e-3,
                       train_print_per_n = 5,
                       valid_per_n = 25,
                       sweep_wandb = None,
                       save_name = None,
                 model_path = './',
                 save_result_path ='./'):

        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        self.G = self.G.to(self.device)
        neg_sampler = Full_Graph_NegSampler(self.G, 1, 'fix_dst', self.device)
        # torch.nn.init.xavier_uniform(self.model.w_rels) # reinitialize decoder

        params = [param for key, param in self.model.prompt.items() if isinstance(param, nn.Parameter)]
        optimizer = torch.optim.AdamW(params + list(self.model.pred.parameters()), lr = learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.8)


        for epoch in range(n_epoch):

            negative_graph = neg_sampler(self.G)

            # pred_score_pos, pred_score_neg, pos_score, neg_score = self.model.forward_prompt(feature_prompt,self.G, negative_graph, pretrain_mode = False, mode = 'train')

            pred_score_pos, pred_score_neg, pos_score, neg_score = self.model(self.G, negative_graph, pretrain_mode=False,mode='train')

            pos_score = torch.cat([pred_score_pos[i] for i in self.dd_etypes])
            neg_score = torch.cat([pred_score_neg[i] for i in self.dd_etypes])

            scores = torch.sigmoid(torch.cat((pos_score, neg_score)).reshape(-1,))
            labels = [1] * len(pos_score) + [0] * len(neg_score)
            loss = F.binary_cross_entropy(scores, torch.Tensor(labels).float().to(self.device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step(loss)

            if self.weight_bias_track:
                self.wandb.log({"Training Loss": loss})

            if epoch % train_print_per_n == 0:
                # training tracking...
                auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc = get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, self.G, True)

                # if self.weight_bias_track:
                #     temp_d = get_wandb_log_dict(auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, "Training")
                #     temp_d.update({"LR": optimizer.param_groups[0]['lr']})
                #     self.wandb.log(temp_d)

                print('Epoch: %d LR: %.5f Loss %.4f, Train Micro AUROC %.4f Train Micro AUPRC %.4f Train Macro AUROC %.4f Train Macro AUPRC %.4f' % (
                    epoch,
                    optimizer.param_groups[0]['lr'],
                    loss.item(),
                    micro_auroc,
                    micro_auprc,
                    macro_auroc,
                    macro_auprc
                ))

                print('----- AUROC Performance in Each Relation -----')
                print_dict(auroc_rel)
                print('----- AUPRC Performance in Each Relation -----')
                print_dict(auprc_rel)
                print('----------------------------------------------')

            del pred_score_pos, pred_score_neg, scores, labels
        with open(save_result_path + "/result.txt", "w") as f:

            # Redirect stdout to the file
            sys.stdout = f
            print('Testing...')

            with torch.no_grad():
                (auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc,
                 macro_auprc), loss, pred_pos, pred_neg = evaluate_fb(self.model, self.g_test_pos,
                                                                      self.g_test_neg, self.G, self.dd_etypes,
                                                                      self.device, True, mode='test')
            print(
                'Testing Loss %.4f Testing Micro AUROC %.4f Testing Micro AUPRC %.4f Testing Macro AUROC %.4f Testing Macro AUPRC %.4f' % (
                    loss,
                    micro_auroc,
                    micro_auprc,
                    macro_auroc,
                    macro_auprc
                ))

            print('----- AUROC Performance in Each Relation -----')
            print_dict(auroc_rel, dd_only=True)
            print('----- AUPRC Performance in Each Relation -----')
            print_dict(auprc_rel, dd_only=True)
            print('----------------------------------------------')

            sys.stdout = sys.__stdout__
            checkpoint_path = os.path.join(save_result_path, f'fintune_model.pth')
            torch.save(self.model.state_dict(), checkpoint_path)

    def predict_all_class(self,model_save_path='./'):


        self.model.load_state_dict(torch.load(model_save_path))
        self.model = self.model.to(self.device)
        self.G = self.G.to(self.device)
        neg_sampler = Full_Graph_NegSampler(self.G, 1, 'fix_dst', self.device)
        # torch.nn.init.xavier_uniform(self.model.w_rels)  # reinitialize decoder
        feature_prompt = featureprompt(self.G, self.model.dgi.prompt, self.model.prompt).cuda()


        negative_graph = neg_sampler(self.G)

        print('Testing...')
        self.best_model = self.model
        (auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc,
         macro_auprc), loss, pred_pos, pred_neg = evaluate_fb_prompt(feature_prompt, self.best_model,
                                                                     self.g_test_pos, self.g_test_neg, self.G,
                                                                     self.dd_etypes, self.device, True,
                                                                     mode='test')

        print(
            'Testing Loss %.4f Testing Micro AUROC %.4f Testing Micro AUPRC %.4f Testing Macro AUROC %.4f Testing Macro AUPRC %.4f' % (
                loss,
                micro_auroc,
                micro_auprc,
                macro_auroc,
                macro_auprc
            ))
        print('----- AUROC Performance in Each Relation -----')
        print_dict(auroc_rel, dd_only=True)
        print('----- AUPRC Performance in Each Relation -----')
        print_dict(auprc_rel, dd_only=True)
        print('----------------------------------------------')

