from tqdm import tqdm
import logging
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)



class Trainer:
    def __init__(self,detectron, pooler, selector, ce_loss, triple_loss, optimizer=None,scheduler=None, save_path= './'):
        self.detectron = detectron 
        self.pooler = pooler 
        self.optimizer = optimizer 
        self.selector = selector 
        self.ce_loss = ce_loss 
        self.triple_loss = triple_loss
        self.save_path = save_path
        self.scheduler =scheduler
     
    
    def train(self, train_loader, eval_loader, model,  max_iters, eval_period = 5000):

        self.detectron.eval()
        model.train()

        best_loss, total_loss, total_correct, total_num, data_bar = 10000, 0, 0, 0,  range(max_iters)
        loader_iter  =  iter(train_loader)

        # no max iter
        for i in data_bar:
            batched_inputs = next(loader_iter)

            with torch.no_grad():
                features, pair_labels, _, is_empty = self.pooler.batches(batched_inputs)

            if is_empty or len(features) < 2:
                continue

            # Clear gradients
            self.optimizer.zero_grad()  

            cgd, logit_for_class = model(features)

            # label smoothing loss
            loss_c = self.ce_loss(logit_for_class, pair_labels)

            # triple loss
            anchor, pos, neg = self.selector(cgd, pair_labels)
            loss_f = self.triple_loss(anchor, pos, neg)

            # total loss
            loss_ = loss_c + loss_f
            loss_.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            self.optimizer.step()
            # init grad
            model.zero_grad()

            pred = torch.max(logit_for_class, dim=-1)[1]
            total_loss += loss_.item() * features.size(0)
            total_correct += torch.sum(pred == pair_labels).item()
            total_num += features.size(0)
            

            
            self.scheduler.step()
            
            
            if (i+1) % 20 ==0:

                logging.info('Train Iter {}/{} - Loss:{:.4f} - cls loss:{:.4f} - triple loss:{:.4f} - Acc:{:.2f}'
                                             .format(i, 
                                                     max_iters,
                                                     loss_c/ total_num,
                                                     loss_f/ total_num ,
                                                     total_loss / total_num,
                                                     total_correct / total_num * 100))
            if (i+1) % eval_period ==0:
                
                eval_loss, eval_cls_loss, eval_triple_loss, eval_acc = self.evaluate(eval_loader, model)

                logging.info('Evaluated - Loss:{:.4f} - cls loss:{:.4f} - triple loss:{:.4f} - Acc:{:.2f}% '
                                     .format(eval_loss, eval_cls_loss, eval_triple_loss, eval_acc))

                if best_loss >  eval_loss:
                    logging.info(f'saved better model...from({best_loss}) to ({eval_loss})')
                    torch.save(model, self.save_path)
                    best_loss = eval_loss 
                    


    def evaluate(self, eval_loader, model, device ='cpu'):
        self.detectron.eval()
        model.eval()

        total_loss, total_correct, total_num, data_bar = 0, 0, 0, tqdm(eval_loader)

        with torch.no_grad():

            for batched_inputs in data_bar:
                features, pair_labels, _, is_empty = self.pooler.batches(batched_inputs)

                if is_empty:
                    continue

                cgd, logit_for_class = model(features)

                # label smoothing loss
                loss_c = self.ce_loss(logit_for_class, pair_labels)

                # triple loss
                anchor, pos, neg = self.selector(cgd, pair_labels)
                loss_f =self.triple_loss(anchor, pos, neg)

                # total loss
                loss_ = loss_c + loss_f

                pred = torch.argmax(logit_for_class, dim=-1)
                total_loss += loss_.item() * features.size(0)
                total_correct += torch.sum(pred == pair_labels).item()
                total_num += features.size(0)

        return loss_c/ total_num, loss_f/ total_num , total_loss / total_num, total_correct / total_num * 100