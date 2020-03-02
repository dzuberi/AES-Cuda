namespace gui
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.button1 = new System.Windows.Forms.Button();
            this.lblHelloWorld = new System.Windows.Forms.Label();
            this.inputLabel = new System.Windows.Forms.Label();
            this.button2 = new System.Windows.Forms.Button();
            this.outputLabel = new System.Windows.Forms.Label();
            this.button4 = new System.Windows.Forms.Button();
            this.threadsBox = new System.Windows.Forms.TextBox();
            this.threadsLabel = new System.Windows.Forms.Label();
            this.radioEncrypt = new System.Windows.Forms.RadioButton();
            this.radioDecrypt = new System.Windows.Forms.RadioButton();
            this.checkBox1 = new System.Windows.Forms.CheckBox();
            this.button3 = new System.Windows.Forms.Button();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.keyTextBox = new System.Windows.Forms.TextBox();
            this.key32 = new System.Windows.Forms.RadioButton();
            this.key16 = new System.Windows.Forms.RadioButton();
            this.groupBox2.SuspendLayout();
            this.SuspendLayout();
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(634, 374);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(75, 23);
            this.button1.TabIndex = 0;
            this.button1.Text = "Click to start";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // lblHelloWorld
            // 
            this.lblHelloWorld.Location = new System.Drawing.Point(584, 400);
            this.lblHelloWorld.Name = "lblHelloWorld";
            this.lblHelloWorld.Size = new System.Drawing.Size(179, 19);
            this.lblHelloWorld.TabIndex = 1;
            this.lblHelloWorld.Text = "Not started";
            this.lblHelloWorld.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // inputLabel
            // 
            this.inputLabel.Location = new System.Drawing.Point(6, 117);
            this.inputLabel.Name = "inputLabel";
            this.inputLabel.Size = new System.Drawing.Size(179, 243);
            this.inputLabel.TabIndex = 3;
            this.inputLabel.Text = "No Input File Chosen";
            this.inputLabel.TextAlign = System.Drawing.ContentAlignment.TopCenter;
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(29, 91);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(129, 23);
            this.button2.TabIndex = 2;
            this.button2.Text = "Choose Input File";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // outputLabel
            // 
            this.outputLabel.ImageAlign = System.Drawing.ContentAlignment.TopCenter;
            this.outputLabel.Location = new System.Drawing.Point(217, 117);
            this.outputLabel.Name = "outputLabel";
            this.outputLabel.Size = new System.Drawing.Size(179, 243);
            this.outputLabel.TabIndex = 5;
            this.outputLabel.Text = "No output file chosen, using (input file name).aes";
            this.outputLabel.TextAlign = System.Drawing.ContentAlignment.TopCenter;
            // 
            // button4
            // 
            this.button4.Location = new System.Drawing.Point(240, 91);
            this.button4.Name = "button4";
            this.button4.Size = new System.Drawing.Size(129, 23);
            this.button4.TabIndex = 6;
            this.button4.Text = "Choose Output File";
            this.button4.UseVisualStyleBackColor = true;
            this.button4.Click += new System.EventHandler(this.button4_Click);
            // 
            // threadsBox
            // 
            this.threadsBox.Location = new System.Drawing.Point(615, 212);
            this.threadsBox.Name = "threadsBox";
            this.threadsBox.Size = new System.Drawing.Size(100, 20);
            this.threadsBox.TabIndex = 7;
            this.threadsBox.TextChanged += new System.EventHandler(this.textBox1_TextChanged);
            // 
            // threadsLabel
            // 
            this.threadsLabel.ImageAlign = System.Drawing.ContentAlignment.TopCenter;
            this.threadsLabel.Location = new System.Drawing.Point(615, 235);
            this.threadsLabel.Name = "threadsLabel";
            this.threadsLabel.Size = new System.Drawing.Size(100, 62);
            this.threadsLabel.TabIndex = 8;
            this.threadsLabel.Text = "number of threads (default 1)";
            this.threadsLabel.TextAlign = System.Drawing.ContentAlignment.TopCenter;
            // 
            // radioEncrypt
            // 
            this.radioEncrypt.AutoSize = true;
            this.radioEncrypt.Checked = true;
            this.radioEncrypt.Location = new System.Drawing.Point(615, 155);
            this.radioEncrypt.Name = "radioEncrypt";
            this.radioEncrypt.Size = new System.Drawing.Size(61, 17);
            this.radioEncrypt.TabIndex = 9;
            this.radioEncrypt.TabStop = true;
            this.radioEncrypt.Text = "Encrypt";
            this.radioEncrypt.UseVisualStyleBackColor = true;
            // 
            // radioDecrypt
            // 
            this.radioDecrypt.AutoSize = true;
            this.radioDecrypt.Location = new System.Drawing.Point(615, 178);
            this.radioDecrypt.Name = "radioDecrypt";
            this.radioDecrypt.Size = new System.Drawing.Size(62, 17);
            this.radioDecrypt.TabIndex = 10;
            this.radioDecrypt.Text = "Decrypt";
            this.radioDecrypt.UseVisualStyleBackColor = true;
            // 
            // checkBox1
            // 
            this.checkBox1.AutoSize = true;
            this.checkBox1.Location = new System.Drawing.Point(615, 280);
            this.checkBox1.Name = "checkBox1";
            this.checkBox1.Size = new System.Drawing.Size(173, 17);
            this.checkBox1.TabIndex = 11;
            this.checkBox1.Text = "cuda - overrides thread number";
            this.checkBox1.UseVisualStyleBackColor = true;
            this.checkBox1.CheckedChanged += new System.EventHandler(this.checkBox1_CheckedChanged);
            // 
            // button3
            // 
            this.button3.Location = new System.Drawing.Point(163, 59);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(75, 23);
            this.button3.TabIndex = 12;
            this.button3.Text = "Clear files";
            this.button3.UseVisualStyleBackColor = true;
            this.button3.Click += new System.EventHandler(this.button3_Click);
            // 
            // groupBox1
            // 
            this.groupBox1.Location = new System.Drawing.Point(609, 135);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(179, 71);
            this.groupBox1.TabIndex = 13;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Encrypt/decrypt";
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.keyTextBox);
            this.groupBox2.Controls.Add(this.key32);
            this.groupBox2.Controls.Add(this.key16);
            this.groupBox2.Location = new System.Drawing.Point(609, 23);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(179, 106);
            this.groupBox2.TabIndex = 14;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Key configuration";
            // 
            // keyTextBox
            // 
            this.keyTextBox.Location = new System.Drawing.Point(6, 65);
            this.keyTextBox.Name = "keyTextBox";
            this.keyTextBox.Size = new System.Drawing.Size(100, 20);
            this.keyTextBox.TabIndex = 15;
            this.keyTextBox.TextChanged += new System.EventHandler(this.keyTextBox_TextChanged);
            // 
            // key32
            // 
            this.key32.AutoSize = true;
            this.key32.Checked = true;
            this.key32.Location = new System.Drawing.Point(6, 42);
            this.key32.Name = "key32";
            this.key32.Size = new System.Drawing.Size(80, 17);
            this.key32.TabIndex = 1;
            this.key32.TabStop = true;
            this.key32.Text = "32 byte key";
            this.key32.UseVisualStyleBackColor = true;
            this.key32.CheckedChanged += new System.EventHandler(this.key32_CheckedChanged);
            // 
            // key16
            // 
            this.key16.AutoSize = true;
            this.key16.Location = new System.Drawing.Point(6, 19);
            this.key16.Name = "key16";
            this.key16.Size = new System.Drawing.Size(80, 17);
            this.key16.TabIndex = 0;
            this.key16.Text = "16 byte key";
            this.key16.UseVisualStyleBackColor = true;
            this.key16.CheckedChanged += new System.EventHandler(this.key16_CheckedChanged);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.button3);
            this.Controls.Add(this.checkBox1);
            this.Controls.Add(this.radioDecrypt);
            this.Controls.Add(this.radioEncrypt);
            this.Controls.Add(this.threadsLabel);
            this.Controls.Add(this.threadsBox);
            this.Controls.Add(this.button4);
            this.Controls.Add(this.outputLabel);
            this.Controls.Add(this.inputLabel);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.lblHelloWorld);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.groupBox1);
            this.Name = "Form1";
            this.Text = "AES Encryptor/Decryptor";
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Label lblHelloWorld;
        private System.Windows.Forms.Label inputLabel;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.Label outputLabel;
        private System.Windows.Forms.Button button4;
        private System.Windows.Forms.TextBox threadsBox;
        private System.Windows.Forms.Label threadsLabel;
        private System.Windows.Forms.RadioButton radioEncrypt;
        private System.Windows.Forms.RadioButton radioDecrypt;
        private System.Windows.Forms.CheckBox checkBox1;
        private System.Windows.Forms.Button button3;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.RadioButton key32;
        private System.Windows.Forms.RadioButton key16;
        private System.Windows.Forms.TextBox keyTextBox;
    }
}

