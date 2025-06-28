import torch
import torch.nn as nn

class AriaXaTModel(nn.Module):
    def __init__(self, input_features, hidden_size=128, num_layers=2, output_classes=3, dropout_rate=0.2):
        super(AriaXaTModel, self).__init__()

        # CNN for feature extraction from each time step
        # Input: (batch_size, seq_len, input_features)
        # Permute to (batch_size, input_features, seq_len) for Conv1d
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=1), # kernel_size=1 processes each timestep independently
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(64, 32, kernel_size=1), # Reduce dimensions further
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # LSTM layer
        # Input to LSTM: (batch_size, seq_len, conv_output_features)
        self.lstm = nn.LSTM(input_size=32,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)

        # Attention Mechanism
        self.attention_linear = nn.Linear(hidden_size, 1) # Maps LSTM output to a single score
        
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
        
        # Apply Conv1d
        # Permute x to (batch_size, input_features, seq_len) for Conv1d
        x_conv = self.conv1d(x.permute(0, 2, 1))
        # Permute back to (batch_size, seq_len, conv_output_features) for LSTM
        x_conv = x_conv.permute(0, 2, 1) # x_conv shape: (batch_size, seq_len, 32)
        
        # Apply LSTM
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x_conv) 
        
        # Apply Attention
        # attention_scores shape: (batch_size, seq_len, 1)
        attention_scores = self.attention_linear(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1) # Softmax over sequence length
        
        # Apply attention weights to LSTM output
        # context_vector shape: (batch_size, hidden_size)
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Pass context vector to classifier and regressor
        classification_output = self.classifier(context_vector)
        regression_output = self.regressor(context_vector)
        
        return classification_output, regression_output.squeeze(1) # Squeeze for single regression value 