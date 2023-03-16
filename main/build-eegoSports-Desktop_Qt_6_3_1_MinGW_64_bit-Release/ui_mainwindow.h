/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 6.3.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindowClass
{
public:
    QAction *actionQuit_2;
    QAction *actionLoad_Configuration;
    QAction *actionSave_Configuration;
    QAction *actionQuit;
    QWidget *centralWidget;
    QHBoxLayout *horizontalLayout_2;
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox;
    QFormLayout *formLayout;
    QLabel *Cap_ID_Label;
    QComboBox *Cap_ID;
    QLabel *SR_Label;
    QComboBox *samplingRate;
    QLabel *EEG_Range_Label;
    QComboBox *EEG_Range;
    QLabel *BIP_Range_Label;
    QComboBox *BIP_Range;
    QLabel *EEG_Mask_Label;
    QLineEdit *EEG_LineEdit;
    QLabel *BIP_Channel_Label;
    QLineEdit *BIP_LineEdit;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer;
    QPushButton *linkButton;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindowClass)
    {
        if (MainWindowClass->objectName().isEmpty())
            MainWindowClass->setObjectName(QString::fromUtf8("MainWindowClass"));
        MainWindowClass->resize(288, 294);
        actionQuit_2 = new QAction(MainWindowClass);
        actionQuit_2->setObjectName(QString::fromUtf8("actionQuit_2"));
        actionLoad_Configuration = new QAction(MainWindowClass);
        actionLoad_Configuration->setObjectName(QString::fromUtf8("actionLoad_Configuration"));
        actionSave_Configuration = new QAction(MainWindowClass);
        actionSave_Configuration->setObjectName(QString::fromUtf8("actionSave_Configuration"));
        actionQuit = new QAction(MainWindowClass);
        actionQuit->setObjectName(QString::fromUtf8("actionQuit"));
        centralWidget = new QWidget(MainWindowClass);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        horizontalLayout_2 = new QHBoxLayout(centralWidget);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        formLayout = new QFormLayout(groupBox);
        formLayout->setSpacing(6);
        formLayout->setContentsMargins(11, 11, 11, 11);
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        formLayout->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
        Cap_ID_Label = new QLabel(groupBox);
        Cap_ID_Label->setObjectName(QString::fromUtf8("Cap_ID_Label"));

        formLayout->setWidget(0, QFormLayout::LabelRole, Cap_ID_Label);

        Cap_ID = new QComboBox(groupBox);
        Cap_ID->addItem(QString());
        Cap_ID->addItem(QString());
        Cap_ID->addItem(QString());
        Cap_ID->addItem(QString());
        Cap_ID->addItem(QString());
        Cap_ID->addItem(QString());
        Cap_ID->setObjectName(QString::fromUtf8("Cap_ID"));

        formLayout->setWidget(0, QFormLayout::FieldRole, Cap_ID);

        SR_Label = new QLabel(groupBox);
        SR_Label->setObjectName(QString::fromUtf8("SR_Label"));

        formLayout->setWidget(1, QFormLayout::LabelRole, SR_Label);

        samplingRate = new QComboBox(groupBox);
        samplingRate->addItem(QString());
        samplingRate->addItem(QString());
        samplingRate->addItem(QString());
        samplingRate->addItem(QString());
        samplingRate->addItem(QString());
        samplingRate->addItem(QString());
        samplingRate->addItem(QString());
        samplingRate->addItem(QString());
        samplingRate->addItem(QString());
        samplingRate->addItem(QString());
        samplingRate->addItem(QString());
        samplingRate->addItem(QString());
        samplingRate->setObjectName(QString::fromUtf8("samplingRate"));

        formLayout->setWidget(1, QFormLayout::FieldRole, samplingRate);

        EEG_Range_Label = new QLabel(groupBox);
        EEG_Range_Label->setObjectName(QString::fromUtf8("EEG_Range_Label"));

        formLayout->setWidget(2, QFormLayout::LabelRole, EEG_Range_Label);

        EEG_Range = new QComboBox(groupBox);
        EEG_Range->addItem(QString());
        EEG_Range->addItem(QString());
        EEG_Range->addItem(QString());
        EEG_Range->setObjectName(QString::fromUtf8("EEG_Range"));

        formLayout->setWidget(2, QFormLayout::FieldRole, EEG_Range);

        BIP_Range_Label = new QLabel(groupBox);
        BIP_Range_Label->setObjectName(QString::fromUtf8("BIP_Range_Label"));

        formLayout->setWidget(3, QFormLayout::LabelRole, BIP_Range_Label);

        BIP_Range = new QComboBox(groupBox);
        BIP_Range->addItem(QString());
        BIP_Range->addItem(QString());
        BIP_Range->addItem(QString());
        BIP_Range->addItem(QString());
        BIP_Range->setObjectName(QString::fromUtf8("BIP_Range"));

        formLayout->setWidget(3, QFormLayout::FieldRole, BIP_Range);

        EEG_Mask_Label = new QLabel(groupBox);
        EEG_Mask_Label->setObjectName(QString::fromUtf8("EEG_Mask_Label"));

        formLayout->setWidget(4, QFormLayout::LabelRole, EEG_Mask_Label);

        EEG_LineEdit = new QLineEdit(groupBox);
        EEG_LineEdit->setObjectName(QString::fromUtf8("EEG_LineEdit"));

        formLayout->setWidget(4, QFormLayout::FieldRole, EEG_LineEdit);

        BIP_Channel_Label = new QLabel(groupBox);
        BIP_Channel_Label->setObjectName(QString::fromUtf8("BIP_Channel_Label"));

        formLayout->setWidget(5, QFormLayout::LabelRole, BIP_Channel_Label);

        BIP_LineEdit = new QLineEdit(groupBox);
        BIP_LineEdit->setObjectName(QString::fromUtf8("BIP_LineEdit"));

        formLayout->setWidget(5, QFormLayout::FieldRole, BIP_LineEdit);


        verticalLayout->addWidget(groupBox);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        linkButton = new QPushButton(centralWidget);
        linkButton->setObjectName(QString::fromUtf8("linkButton"));

        horizontalLayout->addWidget(linkButton);


        verticalLayout->addLayout(horizontalLayout);


        horizontalLayout_2->addLayout(verticalLayout);

        MainWindowClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindowClass);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 288, 22));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QString::fromUtf8("menuFile"));
        MainWindowClass->setMenuBar(menuBar);
        statusBar = new QStatusBar(MainWindowClass);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        MainWindowClass->setStatusBar(statusBar);

        menuBar->addAction(menuFile->menuAction());
        menuFile->addAction(actionLoad_Configuration);
        menuFile->addAction(actionSave_Configuration);
        menuFile->addAction(actionQuit);

        retranslateUi(MainWindowClass);

        samplingRate->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindowClass);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindowClass)
    {
        MainWindowClass->setWindowTitle(QCoreApplication::translate("MainWindowClass", "eegApp", nullptr));
        actionQuit_2->setText(QCoreApplication::translate("MainWindowClass", "Quit", nullptr));
        actionLoad_Configuration->setText(QCoreApplication::translate("MainWindowClass", "Load Configuration", nullptr));
        actionSave_Configuration->setText(QCoreApplication::translate("MainWindowClass", "Save Configuration", nullptr));
        actionQuit->setText(QCoreApplication::translate("MainWindowClass", "Quit", nullptr));
        groupBox->setTitle(QCoreApplication::translate("MainWindowClass", "Device Settings", nullptr));
        Cap_ID_Label->setText(QCoreApplication::translate("MainWindowClass", "CAP ID", nullptr));
        Cap_ID->setItemText(0, QCoreApplication::translate("MainWindowClass", "Music SSVEP (8 ch)", nullptr));
        Cap_ID->setItemText(1, QCoreApplication::translate("MainWindowClass", "CA-203  (32 ch)", nullptr));
        Cap_ID->setItemText(2, QCoreApplication::translate("MainWindowClass", "CA-208  (64 ch)", nullptr));
        Cap_ID->setItemText(3, QCoreApplication::translate("MainWindowClass", "CA-209  (128 ch)", nullptr));
        Cap_ID->setItemText(4, QCoreApplication::translate("MainWindowClass", " 64   - No Labels", nullptr));
        Cap_ID->setItemText(5, QCoreApplication::translate("MainWindowClass", "128 - No Labels", nullptr));

        SR_Label->setText(QCoreApplication::translate("MainWindowClass", "Sampling Rate [Hz]", nullptr));
        samplingRate->setItemText(0, QCoreApplication::translate("MainWindowClass", "500", nullptr));
        samplingRate->setItemText(1, QCoreApplication::translate("MainWindowClass", "512", nullptr));
        samplingRate->setItemText(2, QCoreApplication::translate("MainWindowClass", "1000", nullptr));
        samplingRate->setItemText(3, QCoreApplication::translate("MainWindowClass", "1024", nullptr));
        samplingRate->setItemText(4, QCoreApplication::translate("MainWindowClass", "2000", nullptr));
        samplingRate->setItemText(5, QCoreApplication::translate("MainWindowClass", "2048", nullptr));
        samplingRate->setItemText(6, QCoreApplication::translate("MainWindowClass", "4000", nullptr));
        samplingRate->setItemText(7, QCoreApplication::translate("MainWindowClass", "4096", nullptr));
        samplingRate->setItemText(8, QCoreApplication::translate("MainWindowClass", "8000", nullptr));
        samplingRate->setItemText(9, QCoreApplication::translate("MainWindowClass", "8192", nullptr));
        samplingRate->setItemText(10, QCoreApplication::translate("MainWindowClass", "16000", nullptr));
        samplingRate->setItemText(11, QCoreApplication::translate("MainWindowClass", "16384", nullptr));

#if QT_CONFIG(tooltip)
        samplingRate->setToolTip(QCoreApplication::translate("MainWindowClass", "Sampling rate to use.", nullptr));
#endif // QT_CONFIG(tooltip)
        EEG_Range_Label->setText(QCoreApplication::translate("MainWindowClass", "EEG signal range [mV]", nullptr));
        EEG_Range->setItemText(0, QCoreApplication::translate("MainWindowClass", "150", nullptr));
        EEG_Range->setItemText(1, QCoreApplication::translate("MainWindowClass", "750", nullptr));
        EEG_Range->setItemText(2, QCoreApplication::translate("MainWindowClass", "1000", nullptr));

        BIP_Range_Label->setText(QCoreApplication::translate("MainWindowClass", "BIP signal range [mV]", nullptr));
        BIP_Range->setItemText(0, QCoreApplication::translate("MainWindowClass", "350", nullptr));
        BIP_Range->setItemText(1, QCoreApplication::translate("MainWindowClass", "700", nullptr));
        BIP_Range->setItemText(2, QCoreApplication::translate("MainWindowClass", "1500", nullptr));
        BIP_Range->setItemText(3, QCoreApplication::translate("MainWindowClass", "4000", nullptr));

        EEG_Mask_Label->setText(QCoreApplication::translate("MainWindowClass", "EEG channel mask", nullptr));
        EEG_LineEdit->setText(QCoreApplication::translate("MainWindowClass", "0x00FF", nullptr));
        BIP_Channel_Label->setText(QCoreApplication::translate("MainWindowClass", "BIP channel mask", nullptr));
        BIP_LineEdit->setText(QCoreApplication::translate("MainWindowClass", "0x000000", nullptr));
        linkButton->setText(QCoreApplication::translate("MainWindowClass", "Link", nullptr));
        menuFile->setTitle(QCoreApplication::translate("MainWindowClass", "File", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindowClass: public Ui_MainWindowClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
