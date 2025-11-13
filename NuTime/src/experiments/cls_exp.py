from .base_exp import *
from src.models.build import get_model
from src.models.encoders.build import get_encoder


class ClassificationExp(BaseExp):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.model = self.build_model()
        self.get_optimizer()
        self.get_scheduler()
        save_model_architecture(config, self.model)

    def build_model(self):
        # get encoder and backbone
        encoder = get_encoder(self.config, self.dataloader_dict['train'].dataset)
        model = get_model(self.config)
        model = nn.Sequential(encoder, model)
        # load pretrained model and freeze backbone
        if self.config.pretrained_model:
            model = self.load_pretrained_model(model)
        if self.config.freeze_backbone:
            model = self.freeze_backbone(model)
        # model parallel
        if self.config.gpu is None:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda(self.config.gpu)

        return model

    def load_pretrained_model(self, model):
        checkpoint = torch.load(self.config.load_checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        model_without_ddp = model.module if hasattr(model, 'module') else model

        # --- Resize positional embeddings if mismatch ---
        if '1.pos_embed' in state_dict:
            pretrained_pos = state_dict['1.pos_embed']
            current_pos = model_without_ddp[1].pos_embed

            if pretrained_pos.shape != current_pos.shape:
                print(f"[Pretrained Adapter] Resizing pos_embed from {pretrained_pos.shape} → {current_pos.shape}")
                import torch.nn.functional as F
                # interpolate along the sequence dimension
                new_pos = F.interpolate(
                    pretrained_pos.transpose(1, 2),
                    size=current_pos.shape[1],
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
                state_dict['1.pos_embed'] = new_pos

        # load remaining parameters
        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        print("Missing / Unexpected keys:", msg)
        return model

    def freeze_backbone(self, model):
        # freeze parameters of backbone (i.e., except the last fc layer)
        self.logger.info('Freezing the parameters of backbone...')
        for name, param in model.named_parameters():
            if name not in ['1.fc.weight', '1.fc.bias']:
                param.requires_grad = False
            elif 'weight' in name:
                param.data.normal_(mean=0., std=0.01)
            elif 'bias' in name:
                param.data.zero_()
        update_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(update_parameters) == 2, f'Only the 2 parameters in fc layer are supposed to update, but get {len(update_parameters)} parameters!'

        return model

    @torch.no_grad()
    def evaluate(self, type):
        iter_time, data_time = AverageMeter(), AverageMeter()
        Losses, Acc1s, Acc5s = AverageMeter(), AverageMeter(), AverageMeter()
        # switch to evaluation mode
        self.model.eval()
        # prediction results
        logits_tensor = torch.tensor([], dtype=torch.float)
        preds_tensor = torch.tensor([], dtype=torch.long)
        trues_tensor = torch.tensor([], dtype=torch.long)
        if torch.cuda.is_available():
            logits_tensor = logits_tensor.cuda(self.config.gpu)
            preds_tensor = preds_tensor.cuda(self.config.gpu)
            trues_tensor = trues_tensor.cuda(self.config.gpu)

        # iterate on evaluation dataset
        torch.cuda.synchronize()
        start_time = time.time()
        for (samples, targets) in self.dataloader_dict[type]:
            B = samples.shape[0]  # batch size
            if torch.cuda.is_available():
                samples = samples.cuda(self.config.gpu)
                targets = targets.cuda(self.config.gpu)

            logits = self.model(samples)
            loss = self.loss(logits, targets)
            Losses.update(loss.item(), B)
            # measure accuracy
            acc1, acc5 = accuracy(logits, targets, topk=(1, min(5, self.config.num_classes // 2)))
            Acc1s.update(acc1, B)
            Acc5s.update(acc5, B)

            # accuracy per class
            _, preds = torch.max(logits, dim=1)
            logits_tensor = torch.cat([logits_tensor, logits.float()], dim=0)
            preds_tensor = torch.cat([preds_tensor, preds.long()], dim=0)
            trues_tensor = torch.cat([trues_tensor, targets.long()], dim=0)

            # measure elapsed time
            torch.cuda.synchronize()
            iter_time.update(time.time() - start_time)
            start_time = time.time()

        preds_tensor = concat_all_gather(preds_tensor)
        trues_tensor = concat_all_gather(trues_tensor)

        # save result
        val_acc = Acc1s.avg
        _, val_mf1 = save_result(self.config, type, logits_tensor, preds_tensor, trues_tensor, verbose=False)

        return val_acc, val_mf1

    def train_batch(self, epoch, idx, batch_data):
        samples, targets = batch_data
        B = samples.shape[0]
        if torch.cuda.is_available():
            samples = samples.cuda(self.config.gpu)
            targets = targets.cuda(self.config.gpu)
        # compute output and loss
        logits = self.model(samples)
        loss = self.loss(logits, targets)
        acc1, acc5 = accuracy(logits, targets, topk=(1, min(5, self.config.num_classes // 2)))

        return B, loss, acc1, acc5
