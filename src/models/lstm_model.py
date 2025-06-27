import torch
import torch.nn as nn

class AriaXaTModel(nn.Module):
    def __init__(self, input_features, hidden_size=128, num_layers=2, output_classes=3, dropout_rate=0.2):
        super(AriaXaTModel, self).__init__()

        # CNN for feature extraction from each time step
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=32,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)

        # Attention Mechanism
        self.attention_linear = nn.Linear(hidden_size, 1)
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_classes)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_features)
        x_conv = self.conv1d(x.permute(0, 2, 1))
        x_conv = x_conv.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_conv)
        attention_scores = self.attention_linear(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)
        classification_output = self.classifier(context_vector)
        regression_output = self.regressor(context_vector)
        return classification_output, regression_output.squeeze(1) 