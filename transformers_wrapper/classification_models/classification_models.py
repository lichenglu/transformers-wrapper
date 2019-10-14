import torch
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm, trange, tqdm_notebook
from transformers import AdamW, WarmupLinearSchedule
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
from transformers import XLNetConfig, XLNetTokenizer, XLNetForSequenceClassification
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import enum


class AvailableClassificationModels(enum.Enum):
    BERT_BASE_UNCASED = 'bert-base-uncased'
    BERT_LARGE_UNCASED = 'bert-large-uncased'
    XLNET_BASE_CASED = 'xlnet-base-cased'
    XLNET_LARGE_CASED = 'xlnet-large-cased'
    ROBERTA_BASE = 'roberta-base'
    ROBERTA_LARGE = 'roberta-large'
    ROBERTA_LARGE_MNLI = 'roberta-large-mnli'


def __config_tokenizer(model_name: AvailableClassificationModels, do_lower_case: bool = True):
    model_name = str(model_name.value)
    tokenizer = None
    if 'bert' in model_name:
        tokenizer = BertTokenizer.from_pretrained(
            model_name, do_lower_case=do_lower_case)
    elif 'xlnet' in model_name:
        tokenizer = XLNetTokenizer.from_pretrained(
            model_name, do_lower_case=do_lower_case)
    elif 'roberta' in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(
            model_name, do_lower_case=do_lower_case)

    return tokenizer


def __config_model(model_name: AvailableClassificationModels, num_labels: int, use_gpu: bool):
    model_name = str(model_name.value)
    model = None
    if 'bert' in model_name:
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=True
        )
    elif 'xlnet' in model_name:
        model = XLNetForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=True
        )
    elif 'roberta' in model_name:
        model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=True
        )

    if use_gpu:
        model.cuda()
    return model


class TransformerModelForClassification(object):
    def __init__(
        self,
        model_name: AvailableClassificationModels,
        num_labels: int,
        do_lower_case: bool
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.device_name = torch.cuda.get_device_name(0)

        self.model_name = model_name
        self.tokenizer = __config_tokenizer(self.model_name, do_lower_case)
        self.model = __config_model(
            self.model_name, num_labels, use_gpu=self.n_gpu > 0)

    def tokenize(self, sentences: [str], max_len: int):
        ''' 
        Tokenize string inputs to fit model's requirements. The function will automatically add special BOS and EOS to each sentence that aligns with the specified model's needs.

        Args: 
            sentences ([str]): A sequence of strings.
            max_len (int): The maximum length that your model intends to take. If a sentence's length is greater than max_len, then the sentence will be truncated, otherwise, it you will be padded.

        Returns: 
            input_ids: A sequence of ids from the model's dictionary.
            attention_masks: A sequence of masks that informs model's attention.

        '''
        tokenizer = self.tokenizer

        sentences = self.add_special_char(sentences)
        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

        print(f"The first tokenized text is {tokenized_texts[0]}")

        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        input_ids = [tokenizer.convert_tokens_to_ids(
            x) for x in tokenized_texts]
        # Pad our input tokens
        input_ids = pad_sequences(
            input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")
        # Create attention masks
        attention_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

        return input_ids, attention_masks

    def transform(self, inputs, labels, masks, batch_size: int):
        '''Transform input_ids, labels, and attention_masks to torch tensors.

        Args: 
            inputs ([int]): A sequence of input_ids, which you can get from self.tokenize.
            labels ([int]): A sequence of your labels. Think of it as Y.
            masks ([int]): A sequence of attention masks, which you can get from self.tokenize.
            batch_size (int): Batch size

        Returns: 
            dataloader: A pytorch dataloader ready for model to use.

        '''

        inputs = torch.tensor(inputs)
        masks = torch.tensor(masks)
        labels = torch.tensor(labels)

        batch_size = batch_size

        data = TensorDataset(inputs, masks, labels)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

        return dataloader

    def prepare_data(self, sentences: [str], labels, max_len: int, batch_size: int):
        '''The combination of tokenize and transform.

        Args: 
            sentences ([str]): A sequence of strings.
            labels ([int]): A sequence of your labels. Think of it as Y.
            max_len (int): The maximum length that your model intends to take. If a sentence's length is greater than max_len, then the sentence will be truncated, otherwise, it you will be padded.
            batch_size (int): Batch size

        Returns: 
            dataloader: A pytorch dataloader ready for model to use.

        '''
        inputs, masks = self.tokenize(sentences, max_len)
        return self.transform(inputs, labels, masks, batch_size)

    def __train(self, train_dataloader, optimizer):

        # Tracking variables
        train_loss_set = []
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Set our model to training mode (as opposed to evaluation mode)
        self.model.train()

        pbar = tqdm_notebook(total=len(train_dataloader),
                             desc="Batch", leave=False)

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            train_loss_set.append(loss.item())

            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

            pbar.set_postfix(batch=step+1, loss=tr_loss/nb_tr_steps)
            pbar.update()

        pbar.close()
        tqdm.write("\nTrain loss: {}".format(tr_loss/nb_tr_steps))
        return train_loss_set

    def __validate(self, dataloader):
            # Put model in evaluation mode to evaluate loss on the validation set
        self.model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                output = self.model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = output[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        tqdm.write("\nValidation Accuracy: {}".format(
            eval_accuracy/nb_eval_steps))

    def finetune(self, train_dataloader, validation_dataloader, optimizer, epochs):
        '''Finetune the pretrained model with training and validation data.

        Args: 
            train_dataloader (tourch.DataLoader): Pytorch dataloader for training, which you can get from prepare_data.
            validation_dataloader (tourch.DataLoader): Pytorch dataloader for validation, which you can get from prepare_data.
            optimizer (transformers.AdamW): An adam optimizer to finetune parameters
            epochs (int): Epoch number

        Returns: 
            training_loss: A sequence of training loss of each batch

        '''

        # Store our loss and accuracy for plotting
        train_loss_set = []

        # trange is a tqdm wrapper around the normal python range
        for _ in trange(epochs, desc="Epoch"):

            # Train
            _train_loss_set = self.__train(train_dataloader, optimizer)
            train_loss_set = [*train_loss_set, *_train_loss_set]

            # Validation
            self.__validate(validation_dataloader)

        return _train_loss_set

    def evaluate(self, dataloader, code_book=None, silence=False):
        '''Finetune the pretrained model with training and validation data.

        Args: 
            dataloader (tourch.DataLoader): Pytorch dataloader for evaluation, which you can get from prepare_data.
            code_book (dict): A dictionary that maps numeric labels to string labels. e.g. pd.factorize.
            silence (bool): Whether you want to print the evaluation metrics or not.

        Returns: 
            flat_true_labels: The true labels from your data.
            flat_predictions: The predicted labels from the model.

        '''

        # Put model in evaluation mode
        self.model.eval()

        # Tracking variables
        predictions, true_labels = [], []

        # Predict
        for batch in dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = outputs[0]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        matthews_set = []

        for i in range(len(true_labels)):
            matthews = matthews_corrcoef(true_labels[i],
                                         np.argmax(predictions[i], axis=1).flatten())
            matthews_set.append(matthews)

        # Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
        flat_predictions = [
            item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        flat_true_labels = [
            item for sublist in true_labels for item in sublist]

        if not silence:
            print(
                f"The matthew corrcoef is {matthews_corrcoef(flat_true_labels, flat_predictions)}")
            print(classification_report(flat_true_labels, flat_predictions))
            if code_book:
                print(code_book)

        return flat_true_labels, flat_predictions

    def predict(self, dataloader):
        '''Finetune the pretrained model with training and validation data.

        Args: 
            dataloader (tourch.DataLoader): Pytorch dataloader for prediction, which you can get from prepare_data.

        Returns: 
            prediction: A 1-D array that contains all the predictions (chosen by the largest probability).

        '''
        # Put model in evaluation mode
        self.model.eval()

        # Tracking variables
        predictions = []

        # Predict
        for batch in dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = outputs[0]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Store predictions and true labels
            predictions.append(logits)

        flat_predictions = [
            item for sublist in predictions for item in sublist]
        return np.argmax(flat_predictions, axis=1).flatten()

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def add_special_char(self, sentences):
        model_name = self.model_name.value
        # We need to add special tokens at the beginning and end of each sentence for Transformer to work properly
        if 'xlnet' in model_name:
            sentences = [sentence + " [SEP] [CLS]" for sentence in sentences]
        elif 'bert' in model_name:
            sentences = ["[CLS] " + sentence +
                         " [SEP]" for sentence in sentences]
        elif 'roberta' in model_name:
            sentences = ["<s> " + sentence + " </s>" for sentence in sentences]

        return sentences
