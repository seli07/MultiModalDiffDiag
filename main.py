import os
import re
from pathlib import Path
from copy import deepcopy
from functools import reduce

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder


class DATA2(Dataset):
    def __init__(
        self,
        dataLocation="./data/",
        debug=False,
    ):
        super().__init__()
        self.debug = debug
        self.dataPath = Path(dataLocation)

        if not self.dataPath.is_dir():
            raise FileNotFoundError("Data directory doesn't exist")
        if debug:
            print("Found data directory")
        self.dataFiles = list(self.dataPath.glob("*.pkl"))
        if not len(self.dataFiles):
            raise FileNotFoundError("No PKLs found in data directory")
        if debug:
            print(f"Found {len(self.dataFiles)} PKL files")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if debug:
            print(f"Using {self.device}")
        self.labelEncoder = LabelEncoder()

        self.IDColumn = "Patient IDX"
        self.categoricalColumns = {
            "PatientAdministrationGenderCode": None,
            "Smoking Status": None,
        }
        self.categEmbedColumns = {
            "SnomedEmbed": [
                "SNOMED Codes 0",
                "SNOMED Codes 1",
                "SNOMED Codes 2",
                "SNOMED Codes 3",
                "SNOMED Codes Other",
            ],
            "ProcedureEmbed": [
                "Procedure Codes 0",
                "Procedure Codes 1",
                "Procedure Codes 2",
                "Procedure Codes 3",
                "Procedure Codes Other",
            ],
        }
        self.ignoreColumns = [
            "ICD-10 Codes 0",
            "ICD-10 Codes 1",
            "ICD-10 Codes Other",
            "ICD-10 Codes 2",
            "ICD-10 Codes 3",
        ]
        self.cliNotesColumns = [
            "Projected_Med_Embeddings",
            "Projected_Note_Embeddings",
        ]
        self.yVar = [
            "Chronic kidney disease all stages (1 through 5)",
            "Acute Myocardial Infarction",
            "Hypertension Pulmonary hypertension",
            "Ischemic Heart Disease",
        ]
        self.yVarList = {
            "Diabetes": [
                "Type 1 Diabetes",
                "Type II Diabetes",
            ]
        }
        self.continuousColumns = [
            "PatientBirthDateTime",
            "Systolic Blood Pressure",
            "Diastolic Blood Pressure",
            "Body Weight",
            "Body Height",
            "BMI",
            "Body Temperature",
            "Heart Rate",
            "Oxygen Saturation",
            "Respiratory Rate",
            "Hemoglobin A1C",
            "Blood Urea Nitrogen",
            "Bilirubin lab test",
            "Troponin Lab Test",
            "Ferritin",
            "Glucose Tolerance Testing",
            "Cerebral Spinal Fluid (CSF) Analysis",
            "Arterial Blood Gas",
            "Comprehensive Metabolic Panel",
            "Chloride  Urine",
            "Calcium in Blood  Serum or Plasma",
            "Magnesium in Blood  Serum or Plasma",
            "Magnesium in Urine",
            #"Chloride  Blood  Serum or Plasma",
            "Creatinine  Urine",
            # The following columns have been commented out (or renamed) based on your adjustments:
            # "Creatinine  Blood  Serum or Plasma",
            # "Phosphate Blood  Serum or Plasma",
            "Coagulation Assay",
            "Complete Blood Count",
            # "Creatine Kinase Blood  Serum or Plasma",
            "D Dimer Test",
            # "Electrolytes Panel Blood  Serum or Plasma",
            # "Inflammatory Markers (CRP) Blood  Serum or Plasma",
            # "Lipid Serum or Plasma",
            "Sputum Culture",
            "Urine Collection 24 Hours",
        ]

        self.contData = []
        self.categData = []
        self.clinData = []
        self.labels = []
        self.dataLen = 0
        maxRows = 0
        self.labelEncoder = LabelEncoder()

        # Load ClinicalBERT model and tokenizer
        self.clinicalbert_model_name = "emilyalsentzer/Bio_ClinicalBERT"
        self.clinicalbert_model = AutoModel.from_pretrained(self.clinicalbert_model_name).to(self.device)
        self.clinicalbert_tokenizer = AutoTokenizer.from_pretrained(self.clinicalbert_model_name)

        if debug:
            print("Pre Processing Data...")
        # First pass to fit categorical encoders
        initCategData = []
        for f in tqdm(self.dataFiles, desc="Loading categorical data"):
            df = pd.read_pickle(f)
            initCategData.append(df[list(self.categoricalColumns.keys())].astype("string"))
        initCategData = pd.concat(initCategData, ignore_index=True)
        for categLabel in self.categoricalColumns:
            self.categoricalColumns[categLabel] = LabelEncoder().fit(initCategData[categLabel])
        # Process each file
        for f in tqdm(self.dataFiles, desc="Processing files"):
            df = pd.read_pickle(f)
            patientIDXs = df[self.IDColumn].unique()
            for patientID in tqdm(patientIDXs, desc="Processing patients", leave=False):
                contStack = []
                categStack = []
                clinStack = []
                patientRows = df.loc[df[self.IDColumn] == patientID]
                patientRows = patientRows.ffill().bfill().fillna(0)
                contData = patientRows[self.continuousColumns]
                contData = contData.reset_index(drop=True)
                clinicalEmbeddings = patientRows[self.cliNotesColumns]
                # Process categorical embedding columns
                for categEmbed in self.categEmbedColumns:
                    subset = patientRows[self.categEmbedColumns[categEmbed]].reset_index(drop=True)
                    newData = []
                    subset = subset.astype("string")
                    for i in range(len(subset)):
                        newData.append(self.genEmbeddings("".join(subset.loc[i].to_list())))
                    clinicalEmbeddings[categEmbed] = newData
                clinicalEmbeddings = clinicalEmbeddings.to_numpy()
                categData = patientRows[list(self.categoricalColumns.keys())].to_numpy()
                for i, d in enumerate(self.categoricalColumns):
                    categData[:, i] = self.categoricalColumns[d].transform(categData[:, i])
                for col in contData.columns:
                    contData[col] = pd.to_numeric(contData[col])
                contData = (((contData - contData.mean()) / (contData.std() + 1e-100))
                            .fillna(0)
                            .to_numpy())
                assert len(contData) == len(categData) == len(clinicalEmbeddings), (
                    f"Shape mismatch for patient {patientID}: "
                    f"{len(contData)}, {len(categData)}, {len(clinicalEmbeddings)}"
                )
                labels = patientRows[self.yVar]
                for col in self.yVarList:
                    orValues = reduce(lambda a, b: a | b, patientRows[self.yVarList[col]].T.to_numpy())
                    labels.loc[:, col] = orValues
                labels = (labels.reset_index(drop=True)
                          .replace("", np.nan)
                          .fillna(0)
                          .astype(int)
                          .to_numpy())
                for dRow in range(len(labels)):
                    contStack.append(contData[dRow])
                    categStack.append(categData[dRow])
                    clinStack.append(np.stack(clinicalEmbeddings[dRow]))
                    if np.any(labels[dRow]):
                        self.contData.append(np.stack(deepcopy(contStack)))
                        self.categData.append(np.stack(deepcopy(categStack)))
                        self.clinData.append(np.stack(deepcopy(clinStack)))
                        self.labels.append(np.stack(deepcopy(labels[dRow])))
                        self.dataLen += 1
                if self.contData and len(self.contData[-1]) > maxRows:
                    maxRows = len(self.contData[-1])
                del contStack, categStack, clinStack

        if debug:
            print(f"Max roll up Rows: {maxRows}")
        # Pad sequences to maxRows
        for i, d in enumerate(self.contData):
            self.contData[i] = np.vstack([d, np.zeros((maxRows - d.shape[0], *d.shape[1:]), dtype=d.dtype)])
        for i, d in enumerate(self.categData):
            self.categData[i] = np.vstack([d, np.zeros((maxRows - d.shape[0], *d.shape[1:]), dtype=d.dtype)])
        for i, d in enumerate(self.clinData):
            self.clinData[i] = np.vstack([d, np.zeros((maxRows - d.shape[0], *d.shape[1:]), dtype=d.dtype)])

        # Convert lists to tensors and flatten the time dimension
        self.contData = torch.tensor(np.array(self.contData), dtype=torch.float32).reshape((len(self.contData), -1))
        self.categData = torch.tensor(np.array(self.categData).astype(int), dtype=torch.long).reshape((len(self.categData), -1))
        self.clinData = torch.tensor(np.array(self.clinData), dtype=torch.float32).reshape((len(self.clinData), -1))
        self.contDataInputShape = self.contData.shape[-1]
        self.categDataInputShape = self.categData.shape[-1]
        self.clinDataInputShape = self.clinData.shape[-1]
        self.labels = torch.tensor(np.array(self.labels).astype(int), dtype=torch.float32)
        self.labelOutputShape = self.labels.shape[-1]
        if debug:
            print("Done initializing Dataset.")

    def __getitem__(self, index):
        return (
            self.contData[index],
            self.categData[index],
            self.clinData[index],
            self.labels[index],
        )

    def __len__(self):
        return self.dataLen

    def getTotalRowCount(self):
        totalRows = 0
        for dataFile in self.dataFiles:
            totalRows += len(pd.read_pickle(dataFile))
        return totalRows

    def getShapes(self):
        return (
            self.contDataInputShape,
            self.categDataInputShape,
            self.clinDataInputShape,
            self.labelOutputShape,
        )

    def chunkText(self, text, chunkSize=512):
        chunks = []
        idx = 0
        while idx < len(text):
            end = min(idx + chunkSize, len(text))
            chunk = text[idx:end]
            if end < len(text) and not re.match(r"\b\w+\b$", chunk):
                chunk += " " + text[end]
                end += 1
            chunks.append(chunk)
            idx = end
        return chunks

    def genEmbeddings(self, text):
        chunks = self.chunkText(text)
        embeddings = []
        for chunk in chunks:
            encoded_input = self.clinicalbert_tokenizer(
                chunk, return_tensors="pt", padding="max_length", truncation=True
            )
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            with torch.no_grad():
                output = self.clinicalbert_model(**encoded_input)
                last_hidden_state = output.last_hidden_state
                chunk_embedding = torch.mean(last_hidden_state, dim=1)
                embeddings.append(chunk_embedding.cpu().detach().numpy())
        if embeddings:
            return np.mean(np.concatenate(embeddings), axis=0)
        else:
            return np.zeros((768,))


def loadHIEDATA(
    dataDir="./data/",
    trainBatchSize=8,
    testBatchSize=4,
):
    dataset = DATA2(dataLocation=dataDir, debug=True)
    total_length = len(dataset)
    train_length = int(0.8 * total_length)
    test_length = total_length - train_length

    train_dataset, test_dataset = random_split(
        dataset, 
        [train_length, test_length],
        generator=torch.Generator().manual_seed(42)
    )
    trainLoader = DataLoader(train_dataset, batch_size=trainBatchSize, shuffle=True)
    testLoader = DataLoader(test_dataset, batch_size=testBatchSize, shuffle=False)
    return trainLoader, testLoader, dataset.categoricalColumns, dataset.getShapes()


class ModelDD(nn.Module):
    def __init__(self, contFeatureLen, categLen, clinicNotesLen, outputLen):
        super(ModelDD, self).__init__()
        # Continuous branch
        self.contInput = nn.Linear(contFeatureLen, 128)
        self.inputNorm = nn.LayerNorm(128)
        self.attention = nn.MultiheadAttention(128, 4)
        self.lRelu = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.4)
        self.nextLayerNorm = nn.LayerNorm(128)
        self.conv1 = nn.Linear(128, 512)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.next2LayerNorm = nn.LayerNorm(512)
        self.conv2 = nn.Linear(512, 128)

        # Categorical branch
        self.categInput = nn.Linear(categLen, 128)
        self.cat_2_layerNorm = nn.LayerNorm(128)
        self.cat_3_self_attention = nn.MultiheadAttention(128, 4)
        self.cat_4_conv1 = nn.Linear(256, 256)
        self.cat_5_relu = nn.ReLU()
        self.cat_dropout = nn.Dropout(0.4)
        self.cat_branch_layerNorm = nn.LayerNorm(256)
        self.cat_branch_conv2 = nn.Linear(256, 384)
        self.cat_branch_relu = nn.ReLU()
        self.cat_branch_dropout = nn.Dropout(0.3)

        # Clinical branch
        self.cliInput = nn.Linear(clinicNotesLen, 128)
        self.cli_2_layerNorm = nn.LayerNorm(128)
        self.cli_3_selfAtt = nn.MultiheadAttention(128, 4)
        self.cli_4_layerNorm = nn.LayerNorm(128)
        self.cli_5_conv1 = nn.Linear(128, 512)
        self.cli_6_relu = nn.ReLU()
        self.cli_6_1_dropout = nn.Dropout(0.4)
        self.cli_7_layerNorm = nn.LayerNorm(512)
        self.cli_8_conv2 = nn.Linear(512, 128)
        self.cli_9_layerNorm = nn.LayerNorm(384)
        self.cli_10_conv3l = nn.Linear(384, 1536)
        self.cli_11_conv3r = nn.Linear(384, 384)
        self.cli_12_relu = nn.ReLU()
        self.cli_12_1_dropout = nn.Dropout(0.3)
        self.cli_13_conv4 = nn.Linear(1536, 384)

        # Final output
        self.out = nn.Linear(384, outputLen)

    def forward(self, sample):
        contIn, cateIn, clinIn = sample
        contOutput = self.continuousFeaturesForward(contIn)
        ret1, ret2 = self.categoricalFeaturesForward(cateIn)
        res = self.clinicalFeaturesForward(clinIn, ret1, ret2, contOutput)
        res = self.out(res)
        return res

    def continuousFeaturesForward(self, inp):
        x = self.contInput(inp)
        sav0 = x.clone()
        x = self.inputNorm(x)
        # MultiheadAttention expects (L, N, E); unsqueeze sequence dimension
        x = x.unsqueeze(0)
        attn_output, _ = self.attention(x, x, x, need_weights=False)
        x = self.dropout1(self.lRelu(attn_output.squeeze(0)))
        addOutput = x + sav0
        x = self.nextLayerNorm(addOutput)
        x = self.conv1(x)
        x = self.dropout2(self.relu(x))
        x = self.next2LayerNorm(x)
        x = self.conv2(x)
        output = x + addOutput
        return output

    def categoricalFeaturesForward(self, inp):
        xa = self.categInput(inp)
        x = xa.clone()
        x = self.cat_2_layerNorm(x)
        x = x.unsqueeze(0)
        attn_output, _ = self.cat_3_self_attention(x, x, x, need_weights=False)
        x = attn_output.squeeze(0)
        xb = torch.cat([x, xa], dim=1)
        x = self.cat_4_conv1(xb)
        x = self.cat_dropout(self.cat_5_relu(x))
        xBran = self.cat_branch_layerNorm(xb)
        ret1 = xBran + x
        xBran = self.cat_branch_conv2(xBran)
        ret2 = self.cat_branch_dropout(self.cat_branch_relu(xBran))
        return ret1, ret2

    def clinicalFeaturesForward(self, inp, ret1, ret2, contFeat):
        xa = self.cliInput(inp)
        x = xa.clone()
        x = self.cli_2_layerNorm(x)
        x = x.unsqueeze(0)
        attn_output, _ = self.cli_3_selfAtt(x, x, x, need_weights=False)
        x = attn_output.squeeze(0)
        x = x + xa
        xb = self.cli_4_layerNorm(x)
        x = self.cli_5_conv1(xb)
        x = self.cli_6_1_dropout(self.cli_6_relu(x))
        x = self.cli_7_layerNorm(x)
        x = self.cli_8_conv2(x)
        x = x + xb
        xc = torch.cat([x, ret1], dim=1)
        x = self.cli_9_layerNorm(xc)
        xdl = self.cli_10_conv3l(x)
        xdr = self.cli_11_conv3r(x)
        x = self.cli_12_1_dropout(self.cli_12_relu(xdl))
        x = self.cli_13_conv4(x)
        pad_size = max(0, xc.size(1) - contFeat.size(1))
        contFeat_padded = nn.functional.pad(contFeat, (0, pad_size), value=0)
        x = x + xc + contFeat_padded + xdr + ret2
        return x


def train_model(model, train_loader, test_loader, device, epochs=10, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Assuming multi-label classification; adjust loss if necessary
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            cont, categ, clin, labels = batch
            cont = cont.to(device)
            categ = categ.to(device).float()
            clin = clin.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model((cont, categ, clin))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} Testing", leave=False):
                cont, categ, clin, labels = batch
                cont = cont.to(device)
                categ = categ.to(device).float()
                clin = clin.to(device)
                labels = labels.to(device)
                outputs = model((cont, categ, clin))
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch {epoch+1}: Test Loss = {avg_test_loss:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    dataDir = "./data/"
    train_loader, test_loader, categEncoders, shapes = loadHIEDATA(dataDir=dataDir)
    contFeatureLen, categLen, clinicNotesLen, outputLen = shapes

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = ModelDD(contFeatureLen, categLen, clinicNotesLen, outputLen)
    print(model)

    # Start training
    train_model(model, train_loader, test_loader, device, epochs=10, lr=1e-3)
